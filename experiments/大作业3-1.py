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

# 解决中文乱码 + 屏蔽多余警告
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ===================== 1. 数据集路径（沿用你本地路径） =====================
root_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cnews"

train_path = os.path.join(root_dir, "cnews.train.txt")
val_path = os.path.join(root_dir, "cnews.val.txt")
test_path = os.path.join(root_dir, "cnews.test.txt")
vocab_path = os.path.join(root_dir, "cnews.vocab.txt")

# ===================== 2. 超参数优化 =====================
VOCAB_SIZE = 10000      # 扩大词表，覆盖更多词汇
MAX_SEQ_LEN = 150       # 增加序列长度，保留更多文本信息
EMBED_DIM = 128         # 提升词向量维度，增强语义表达
BATCH_SIZE = 128
EPOCHS = 15

# ===================== 3. 加载完整数据 + 分词 =====================
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

# ===================== 4. 文本预处理 =====================
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]
word_index = {word: i+1 for i, word in enumerate(vocab)}

def texts_to_sequences(texts, word_index, max_len):
    sequences = []
    for text in texts:
        seq = [word_index.get(word, 1) for word in text.split()]
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

# ===================== 5. 自定义注意力层 =====================
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

# ===================== 6. 回调函数（防止过拟合 + 动态调学习率） =====================
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
callbacks = [early_stop, reduce_lr]

# ===================== 7. 基础 LSTM 模型（增强版） =====================
print("\n========== 基础 LSTM 模型 ==========")
base_lstm = models.Sequential([
    layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBED_DIM,
        embeddings_regularizer=regularizers.l2(1e-4)
    ),
    layers.LSTM(128, kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

base_lstm.compile(
    optimizer="adam",
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

# ===================== 8. LSTM + 注意力模型（增强版） =====================
print("\n========== LSTM + 注意力模型 ==========")
inputs = layers.Input(shape=(MAX_SEQ_LEN,))
emb = layers.Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBED_DIM,
    embeddings_regularizer=regularizers.l2(1e-4)
)(inputs)
lstm = layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))(emb)
drop = layers.Dropout(0.5)(lstm)
attn_out = Attention()(drop)
outputs = layers.Dense(num_classes, activation="softmax")(attn_out)

attn_model = models.Model(inputs=inputs, outputs=outputs)
attn_model.compile(
    optimizer="adam",
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

# ===================== 9. 绘图（适配提前停止的轮数） =====================
def plot_curves(hist_base, hist_attn):
    # 取两个模型中较短的轮数，保证维度对齐
    min_len = min(len(hist_base.history["accuracy"]), len(hist_attn.history["accuracy"]))
    epochs = range(1, min_len + 1)

    # 截断数据
    base_acc = hist_base.history["accuracy"][:min_len]
    base_val_acc = hist_base.history["val_accuracy"][:min_len]
    base_loss = hist_base.history["loss"][:min_len]
    base_val_loss = hist_base.history["val_loss"][:min_len]

    attn_acc = hist_attn.history["accuracy"][:min_len]
    attn_val_acc = hist_attn.history["val_accuracy"][:min_len]
    attn_loss = hist_attn.history["loss"][:min_len]
    attn_val_loss = hist_attn.history["val_loss"][:min_len]

    plt.figure(figsize=(12, 5))
    # 准确率
    plt.subplot(1, 2, 1)
    plt.plot(epochs, base_acc, "b-", label="基础LSTM 训练准确率")
    plt.plot(epochs, base_val_acc, "b--", label="基础LSTM 验证准确率")
    plt.plot(epochs, attn_acc, "r-", label="LSTM+注意力 训练准确率")
    plt.plot(epochs, attn_val_acc, "r--", label="LSTM+注意力 验证准确率")
    plt.title("模型准确率对比")
    plt.xlabel("迭代轮次")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)

    # 损失
    plt.subplot(1, 2, 2)
    plt.plot(epochs, base_loss, "b-", label="基础LSTM 训练损失")
    plt.plot(epochs, base_val_loss, "b--", label="基础LSTM 验证损失")
    plt.plot(epochs, attn_loss, "r-", label="LSTM+注意力 训练损失")
    plt.plot(epochs, attn_val_loss, "r--", label="LSTM+注意力 验证损失")
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