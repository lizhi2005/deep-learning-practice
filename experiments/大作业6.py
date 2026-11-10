import os
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from gensim.models import Word2Vec
# 环境配置（屏蔽所有tf警告）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")
# 路径
root_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cnews"
train_path = os.path.join(root_dir, "cnews.train.txt")
val_path = os.path.join(root_dir, "cnews.val.txt")
test_path = os.path.join(root_dir, "cnews.test.txt")
vocab_path = os.path.join(root_dir, "cnews.vocab.txt")
#  超参数
VOCAB_SIZE = 15000
MAX_SEQ_LEN = 200
EMBED_DIM = 192
LSTM_UNITS = 160
BATCH_SIZE = 256
EPOCHS = 30
BASE_LR = 2.5e-4
L2_REG = 8e-5
DROPOUT = 0.35
#  加载cnews数据、jieba分词
def load_cnews_data(file_path):
    texts = []
    labels = []
    seg_corpus = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, content = line.split("\t", 1)
            seg = jieba.lcut(content)
            seg_corpus.append(seg)
            seg_content = " ".join(seg)
            texts.append(seg_content)
            labels.append(label)
    return texts, labels, seg_corpus
print("正在加载训练集...")
train_texts, train_labels, train_seg = load_cnews_data(train_path)
print("正在加载验证集...")
val_texts, val_labels, val_seg = load_cnews_data(val_path)
print("正在加载测试集...")
test_texts, test_labels, test_seg = load_cnews_data(test_path)
print(f"训练集样本数：{len(train_texts)}")
print(f"验证集样本数：{len(val_labels)}")
print(f"测试集样本数：{len(test_texts)}")
print(f"类别列表：{sorted(set(train_labels))}")
# 文本序列编码
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]
word_index = {word: i + 1 for i, word in enumerate(vocab)}
def texts_to_sequences(texts, word_dict, max_len):
    sequences = []
    for text in texts:
        seq = [word_dict.get(word, 1) for word in text.split()]
        seq = seq[:max_len]
        sequences.append(seq)
    return pad_sequences(sequences, maxlen=max_len)
x_train = texts_to_sequences(train_texts, word_index, MAX_SEQ_LEN)
x_val = texts_to_sequences(val_texts, word_index, MAX_SEQ_LEN)
x_test = texts_to_sequences(test_texts, word_index, MAX_SEQ_LEN)
# 标签编码独热
le = LabelEncoder()
all_labels = train_labels + val_labels + test_labels
le.fit(all_labels)
y_train_raw = le.transform(train_labels)
y_val_raw = le.transform(val_labels)
y_test_raw = le.transform(test_labels)
num_classes = len(le.classes_)
y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes)
y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes)
print(f"训练集形状：{x_train.shape}，分类数：{num_classes}")
# 样本不平衡：类别权重
from collections import Counter
count_dict = Counter(y_train_raw)
total = len(y_train_raw)
class_weights = {}
for cls, cnt in count_dict.items():
    class_weights[cls] = total / (num_classes * cnt)
#  两种词嵌入矩阵：随机 / Word2Vec预训练
#随机初始化嵌入
rand_emb_matrix = np.random.uniform(-0.25, 0.25, (VOCAB_SIZE, EMBED_DIM))
# 训练Word2Vec预训练向量
all_train_seg = train_seg
w2v = Word2Vec(sentences=all_train_seg, vector_size=EMBED_DIM, window=5, min_count=2, workers=4)
pretrain_emb_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM))
for word, idx in word_index.items():
    if idx >= VOCAB_SIZE:
        continue
    if word in w2v.wv:
        pretrain_emb_matrix[idx] = w2v.wv[word]
# 自定义注意力层
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.att_w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)
    def call(self, inputs):
        score = tf.matmul(tf.reshape(inputs, (-1, inputs.shape[-1])), self.att_w)
        score = tf.reshape(score, (-1, inputs.shape[1]))
        attn_weight = tf.nn.softmax(score)
        attn_weight = tf.expand_dims(attn_weight, axis=-1)
        feat = inputs * attn_weight
        return tf.reduce_sum(feat, axis=1)
# 单层单向LSTM
def build_single_lstm(emb_matrix, trainable_emb=True):
    vocab_len, emb_dim = emb_matrix.shape
    inputs = layers.Input(shape=(MAX_SEQ_LEN,))
    emb = layers.Embedding(
        input_dim=vocab_len,
        output_dim=emb_dim,
        embeddings_initializer=tf.keras.initializers.Constant(emb_matrix),
        mask_zero=False,
        embeddings_regularizer=regularizers.l2(L2_REG),
        trainable=trainable_emb
    )(inputs)
    lstm_out = layers.LSTM(LSTM_UNITS, kernel_regularizer=regularizers.l2(L2_REG))(emb)
    drop1 = layers.Dropout(DROPOUT)(lstm_out)
    dense = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(drop1)
    drop2 = layers.Dropout(DROPOUT)(dense)
    out = layers.Dense(num_classes, activation="softmax")(drop2)
    model = models.Model(inputs, out)
    opt = tf.keras.optimizers.Adam(learning_rate=BASE_LR, clipnorm=1.2, amsgrad=True)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
# 双层BiLSTM
def build_bilstm(emb_matrix, trainable_emb=True):
    vocab_len, emb_dim = emb_matrix.shape
    inputs = layers.Input(shape=(MAX_SEQ_LEN,))
    emb = layers.Embedding(
        input_dim=vocab_len,
        output_dim=emb_dim,
        embeddings_initializer=tf.keras.initializers.Constant(emb_matrix),
        mask_zero=False,
        embeddings_regularizer=regularizers.l2(L2_REG),
        trainable=trainable_emb
    )(inputs)
    bi1 = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG)))(emb)
    bi1 = layers.Dropout(DROPOUT)(bi1)
    bi2 = layers.Bidirectional(layers.LSTM(LSTM_UNITS, kernel_regularizer=regularizers.l2(L2_REG)))(bi1)
    drop1 = layers.Dropout(DROPOUT)(bi2)
    dense = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(drop1)
    drop2 = layers.Dropout(DROPOUT)(dense)
    out = layers.Dense(num_classes, activation="softmax")(drop2)
    model = models.Model(inputs, out)
    opt = tf.keras.optimizers.Adam(learning_rate=BASE_LR, clipnorm=1.2, amsgrad=True)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
# 双层BiLSTM + 注意力
def build_bilstm_attn(emb_matrix, trainable_emb=True):
    vocab_len, emb_dim = emb_matrix.shape
    inputs = layers.Input(shape=(MAX_SEQ_LEN,))
    emb = layers.Embedding(
        input_dim=vocab_len,
        output_dim=emb_dim,
        embeddings_initializer=tf.keras.initializers.Constant(emb_matrix),
        mask_zero=False,
        embeddings_regularizer=regularizers.l2(L2_REG),
        trainable=trainable_emb
    )(inputs)
    bi1 = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG)))(emb)
    bi1 = layers.Dropout(DROPOUT)(bi1)
    bi2 = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG)))(bi1)
    bi2 = layers.Dropout(DROPOUT)(bi2)
    attn_feat = Attention()(bi2)
    drop1 = layers.Dropout(DROPOUT)(attn_feat)
    dense = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(drop1)
    drop2 = layers.Dropout(DROPOUT)(dense)
    out = layers.Dense(num_classes, activation="softmax")(drop2)
    model = models.Model(inputs, out)
    opt = tf.keras.optimizers.Adam(learning_rate=BASE_LR, clipnorm=1.2, amsgrad=True)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
def get_callbacks(name):
    ckpt_path = f"{name}_best.keras"
    ckpt = ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0)
    es = EarlyStopping(monitor="val_accuracy", patience=7, mode="max", restore_best_weights=True, verbose=1)
    lr_decay = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3, min_lr=5e-6, verbose=1)
    return [ckpt, es, lr_decay], ckpt_path
#  训练、评估、绘图、输出指标
def run_exp(model, exp_name, emb_mat, use_attn=False):
    cb, ckpt_path = get_callbacks(exp_name)
    print(f"\n========== 开始训练 {exp_name} ==========")
    hist = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        class_weight=class_weights,
        callbacks=cb,
        verbose=1
    )
    # 加载最优模型，区分是否包含Attention自定义层
    if use_attn:
        best_m = models.load_model(ckpt_path, custom_objects={"Attention": Attention})
    else:
        best_m = models.load_model(ckpt_path)
    test_loss, test_acc = best_m.evaluate(x_test, y_test, verbose=0)
    y_pred_prob = best_m.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test_raw
    print(f"\n【{exp_name}】测试准确率：{test_acc:.4f} | 测试损失：{test_loss:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"{exp_name} 宏F1：{macro_f1:.4f}")
    # 绘制曲线
    epochs = range(len(hist.history["accuracy"]))
    plt.figure(figsize=(11, 4.5))
    plt.subplot(1,2,1)
    plt.plot(epochs, hist.history["accuracy"], label="训练准确率")
    plt.plot(epochs, hist.history["val_accuracy"], label="验证准确率")
    plt.title(f"{exp_name} 准确率曲线")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, hist.history["loss"], label="训练损失")
    plt.plot(epochs, hist.history["val_loss"], label="验证损失")
    plt.title(f"{exp_name} 损失曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{exp_name}_曲线.png", dpi=300)
    plt.show()

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{exp_name} 混淆矩阵")
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.colorbar()
    plt.savefig(f"{exp_name}_混淆矩阵.png", dpi=300)
    plt.show()
    return hist, test_acc, macro_f1
# 全部对照实验
if __name__ == "__main__":
    experiments = [
        ("单向LSTM_随机嵌入", build_single_lstm, rand_emb_matrix, False),
        ("BiLSTM_随机嵌入", build_bilstm, rand_emb_matrix, False),
        ("BiLSTM_注意力_随机嵌入", build_bilstm_attn, rand_emb_matrix, True),
        ("单向LSTM_Word2Vec", build_single_lstm, pretrain_emb_matrix, False),
        ("BiLSTM_Word2Vec", build_bilstm, pretrain_emb_matrix, False),
        ("BiLSTM_注意力_Word2Vec", build_bilstm_attn, pretrain_emb_matrix, True),
    ]
    result = {}
    for name, builder, emb, use_attn_flag in experiments:
        net = builder(emb)
        net.summary()
        h, acc, f1 = run_exp(net, name, emb, use_attn_flag)
        result[name] = {"acc":acc, "macro_f1":f1}
    print("\n" + "="*70)
    print("全部实验汇总结果")
    print("="*70)
    for k,v in result.items():
        print(f"{k:<32} 准确率:{v['acc']:.4f} 宏F1:{v['macro_f1']:.4f}")