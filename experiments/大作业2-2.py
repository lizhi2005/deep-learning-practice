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

# 环境配置：中文显示 + 屏蔽警告
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

# ===================== 1. 路径配置 =====================
root_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cnews"
train_path = os.path.join(root_dir, "cnews.train.txt")
val_path = os.path.join(root_dir, "cnews.val.txt")
test_path = os.path.join(root_dir, "cnews.test.txt")
vocab_path = os.path.join(root_dir, "cnews.vocab.txt")

# ===================== 2. 高阶超参数（针对85%+准确率配置） =====================
VOCAB_SIZE = 12000       # 扩大词表，容纳更多词汇
MAX_SEQ_LEN = 180        # 加长序列，保留完整语义
EMBED_DIM = 160          # 更高维词向量
LSTM_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 3e-4     # 精细调优学习率
L2_REG = 1e-4
DROPOUT_RATE = 0.4

# ===================== 3. 加载数据 + 分词 =====================
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
print(f"所有类别：{sorted(set(train_labels))}")

# ===================== 4. 文本向量化 =====================
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

# 标签独热编码
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

# ===================== 6. 训练回调 & 优化器（核心优化） =====================
# 早停：监控验证损失，耐心值提高，保留最优权重
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
# 学习率动态衰减
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.6,
    patience=3,
    min_lr=1e-5,
    verbose=1
)
callbacks = [early_stop, reduce_lr]

# 自定义Adam + 梯度裁剪，防止梯度爆炸
opt = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    clipnorm=1.0  # 梯度裁剪，稳定训练
)

# ===================== 7. 优化版：双向LSTM + 注意力模型 =====================
print("\n========== 双向LSTM + 注意力模型 ==========")
inputs = layers.Input(shape=(MAX_SEQ_LEN,))

# 词嵌入层 + L2正则
emb = layers.Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBED_DIM,
    embeddings_regularizer=regularizers.l2(L2_REG)
)(inputs)

# 双向LSTM（捕捉前后文，大幅提升特征能力）
bi_lstm = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG))
)(emb)

drop = layers.Dropout(DROPOUT_RATE)(bi_lstm)
attn_out = Attention()(drop)

# 输出层
outputs = layers.Dense(num_classes, activation="softmax")(attn_out)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 开始训练
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# 测试集评估
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nLSTM+注意力 测试准确率：{test_acc:.4f}")

# ===================== 8. 绘制训练曲线 =====================
def plot_curves(hist):
    epochs = range(1, len(hist.history["accuracy"]) + 1)
    plt.figure(figsize=(10, 4))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.history["accuracy"], "r-", label="训练准确率")
    plt.plot(epochs, hist.history["val_accuracy"], "r--", label="验证准确率")
    plt.title("模型准确率变化")
    plt.xlabel("迭代轮次")
    plt.ylabel("准确率")
    plt.legend()
    plt.grid(True)

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.history["loss"], "r-", label="训练损失")
    plt.plot(epochs, hist.history["val_loss"], "r--", label="验证损失")
    plt.title("模型损失变化")
    plt.xlabel("迭代轮次")
    plt.ylabel("损失值")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_curves(history)

# 最终结果输出
print("\n===== 实验最终结果 =====")
print(f"测试集准确率: {test_acc:.4f}")