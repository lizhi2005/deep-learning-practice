import os
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===================== 环境配置 =====================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

# ===================== 路径 =====================
root_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cnews"
train_path = os.path.join(root_dir, "cnews.train.txt")
val_path = os.path.join(root_dir, "cnews.val.txt")
test_path = os.path.join(root_dir, "cnews.test.txt")
vocab_path = os.path.join(root_dir, "cnews.vocab.txt")

# ===================== 高阶超参数（冲90%专用） =====================
VOCAB_SIZE = 15000
MAX_SEQ_LEN = 200
EMBED_DIM = 192
LSTM_UNITS = 160
BATCH_SIZE = 256
EPOCHS = 30
BASE_LR = 2.5e-4
L2_REG = 8e-5
DROPOUT = 0.35

# ===================== 加载数据 + 分词 =====================
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
print(f"验证集样本数：{len(val_labels)}")
print(f"测试集样本数：{len(test_texts)}")
print(f"类别列表：{sorted(set(train_labels))}")

# ===================== 文本序列转换 =====================
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

print(f"训练集形状：{x_train.shape}，分类数：{num_classes}")

# ===================== 自定义注意力层 =====================
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

# ===================== 训练回调函数 =====================
# 保存最优模型
checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
# 早停
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    mode="max",
    restore_best_weights=True,
    verbose=1
)
# 学习率衰减
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.7,
    patience=3,
    min_lr=5e-6,
    verbose=1
)
callbacks = [checkpoint, early_stop, reduce_lr]

# 优化器 + 梯度裁剪
opt = tf.keras.optimizers.Adam(
    learning_rate=BASE_LR,
    clipnorm=1.2,
    amsgrad=True
)

# ===================== 多层双向LSTM + 注意力 主模型 =====================
print("\n========== 多层双向LSTM + 注意力 模型 ==========")
inputs = layers.Input(shape=(MAX_SEQ_LEN,))

# 词嵌入层
emb = layers.Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBED_DIM,
    embeddings_regularizer=regularizers.l2(L2_REG),
    mask_zero=True
)(inputs)

# 第一层双向LSTM
bi_lstm1 = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG))
)(emb)
bi_lstm1 = layers.Dropout(DROPOUT)(bi_lstm1)

# 第二层双向LSTM（加深网络）
bi_lstm2 = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG))
)(bi_lstm1)
bi_lstm2 = layers.Dropout(DROPOUT)(bi_lstm2)

# 注意力层
attn_feat = Attention()(bi_lstm2)

# 全连接层
fc = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(attn_feat)
fc = layers.Dropout(DROPOUT)(fc)

# 输出分类
outputs = layers.Dense(num_classes, activation="softmax")(fc)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 训练
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# 测试集评估
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n【最终结果】测试集准确率：{test_acc:.4f}")

# ===================== 绘制训练曲线 =====================
def plot_curve(hist):
    epochs = range(1, len(hist.history["accuracy"]) + 1)
    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.history["accuracy"], "c-", label="训练准确率")
    plt.plot(epochs, hist.history["val_accuracy"], "orange", linestyle="--", label="验证准确率")
    plt.title("准确率变化曲线")
    plt.xlabel("迭代轮次")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.history["loss"], "c-", label="训练损失")
    plt.plot(epochs, hist.history["val_loss"], "orange", linestyle="--", label="验证损失")
    plt.title("损失变化曲线")
    plt.xlabel("迭代轮次")
    plt.ylabel("损失值")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_curve(history)

print("\n===== 汇总 =====")
print(f"测试准确率: {test_acc:.4f}")