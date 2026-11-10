import os
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 屏蔽警告 + 中文绘图
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 关闭oneDNN警告
tf.get_logger().setLevel("ERROR")

# ===================== 路径 =====================
root_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cnews"
train_path = os.path.join(root_dir, "cnews.train.txt")
val_path = os.path.join(root_dir, "cnews.val.txt")
test_path = os.path.join(root_dir, "cnews.test.txt")
vocab_path = os.path.join(root_dir, "cnews.vocab.txt")

# ===================== 最优超参数 =====================
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 150
EMBED_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 20
LR = 5e-4  # 调低学习率，解决震荡

# ===================== 加载数据 =====================
def load_cnews_data(file_path):
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, content = line.split("\t", 1)
            seg_content = " ".join(jieba.lcut(content))
            texts.append(seg_content)
            labels.append(label)
    return texts, labels

print("正在加载训练集...")
train_texts, train_labels = load_cnews_data(train_path)
print("正在加载验证集...")
val_texts, val_labels = load_cnews_data(val_path)
print("正在加载测试集...")
test_texts, test_labels = load_cnews_data(test_path)

print(f"训练集样本数：{len(train_texts)}")
print(f"验证集样本数：{len(val_texts)}")
print(f"测试集样本数：{len(test_texts)}")
print(f"所有类别：{sorted(set(train_labels))}")

# ===================== 文本向量化 =====================
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]
word_index = {word: i+1 for i, word in enumerate(vocab)}

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

# 标签编码
le = LabelEncoder()
all_labels = train_labels + val_labels + test_labels
le.fit(all_labels)
y_train = le.transform(train_labels)
y_val = le.transform(val_labels)
y_test = le.transform(test_labels)

num_classes = len(le.classes_)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"训练集形状：{x_train.shape}, 标签类别数：{num_classes}")

# ===================== 自定义注意力层 =====================
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.att_weight = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        score = tf.matmul(tf.reshape(inputs, (-1, inputs.shape[-1])), self.att_weight)
        score = tf.reshape(score, (-1, inputs.shape[1]))
        attn_weights = tf.nn.softmax(score)
        attn_weights = tf.expand_dims(attn_weights, axis=-1)
        weighted_feature = inputs * attn_weights
        return tf.reduce_sum(weighted_feature, axis=1)

# ===================== 训练回调函数 =====================
early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=2, min_lr=1e-5, verbose=1)
callbacks = [early_stop, reduce_lr]

# 自定义优化器
opt = tf.keras.optimizers.Adam(learning_rate=LR)

# ===================== 修复版 基础LSTM（核心改动：全局池化 + 完整序列） =====================
print("\n========== 基础 LSTM 模型（优化版） ==========")
base_lstm = models.Sequential([
    layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBED_DIM,
        input_length=MAX_SEQ_LEN,
        embeddings_regularizer=regularizers.l2(1e-4)
    ),
    # 输出完整序列，不再只取最后一步
    layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)),
    layers.GlobalAveragePooling1D(),  # 全局平均池化，聚合全文信息
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

base_lstm.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
base_lstm.summary()

history_base = base_lstm.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

base_loss, base_acc = base_lstm.evaluate(x_test, y_test, verbose=0)
print(f"基础LSTM 测试准确率：{base_acc:.4f}")

# ===================== LSTM + 注意力模型（同步优化） =====================
print("\n========== LSTM + 注意力模型 ==========")
inputs = layers.Input(shape=(MAX_SEQ_LEN,))
emb = layers.Embedding(
    VOCAB_SIZE, EMBED_DIM,
    embeddings_regularizer=regularizers.l2(1e-4)
)(inputs)
lstm = layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))(emb)
drop = layers.Dropout(0.5)(lstm)
attn_out = Attention()(drop)
outputs = layers.Dense(num_classes, activation="softmax")(attn_out)

attn_model = models.Model(inputs=inputs, outputs=outputs)
attn_model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
attn_model.summary()

history_attn = attn_model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

attn_loss, attn_acc = attn_model.evaluate(x_test, y_test, verbose=0)
print(f"LSTM+注意力 测试准确率：{attn_acc:.4f}")

# ===================== 绘图 =====================
def plot_curves(hist_base, hist_attn):
    min_len = min(len(hist_base.history["accuracy"]), len(hist_attn.history["accuracy"]))
    epochs = range(1, min_len + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist_base.history["accuracy"][:min_len], "b-", label="基础LSTM 训练准确率")
    plt.plot(epochs, hist_base.history["val_accuracy"][:min_len], "b--", label="基础LSTM 验证准确率")
    plt.plot(epochs, hist_attn.history["accuracy"][:min_len], "r-", label="LSTM+注意力 训练准确率")
    plt.plot(epochs, hist_attn.history["val_accuracy"][:min_len], "r--", label="LSTM+注意力 验证准确率")
    plt.title("模型准确率对比")
    plt.xlabel("迭代轮次")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist_base.history["loss"][:min_len], "b-", label="基础LSTM 训练损失")
    plt.plot(epochs, hist_base.history["val_loss"][:min_len], "b--", label="基础LSTM 验证损失")
    plt.plot(epochs, hist_attn.history["loss"][:min_len], "r-", label="LSTM+注意力 训练损失")
    plt.plot(epochs, hist_attn.history["val_loss"][:min_len], "r--", label="LSTM+注意力 验证损失")
    plt.title("模型损失对比")
    plt.xlabel("迭代轮次")
    plt.ylabel("损失值")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_curves(history_base, history_attn)

# 结果汇总
print("\n===== 最终结果汇总 =====")
print(f"基础LSTM 测试准确率: {base_acc:.4f}")
print(f"LSTM+注意力 测试准确率: {attn_acc:.4f}")