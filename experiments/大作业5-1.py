import os
import jieba
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

# ===================== 环境配置 =====================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")
# 注释混合精度，解决float16/float32类型冲突
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ===================== 路径 =====================
root_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cnews"
train_path = os.path.join(root_dir, "cnews.train.txt")
val_path = os.path.join(root_dir, "cnews.val.txt")
test_path = os.path.join(root_dir, "cnews.test.txt")
vocab_path = os.path.join(root_dir, "cnews.vocab.txt")

# ===================== 全新高阶超参数（冲85%+专用） =====================
VOCAB_SIZE = 12000    # 减小词表，过滤低频噪声
MAX_SEQ_LEN = 250     # 加长文本，保留更多语义
EMBED_DIM = 256       # 更大词嵌入维度
LSTM_UNITS = 200      # 单层LSTM单元提升
BATCH_SIZE = 256
EPOCHS = 40
BASE_LR = 3e-4
MIN_LR = 1e-6
L2_REG = 5e-5
DROPOUT = 0.4
LABEL_SMOOTH = 0.1    # 标签平滑抑制过拟合
WARMUP_EPOCHS = 3     # 学习率预热轮次

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

# ===================== 优化词表映射：截断低频词 =====================
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_all = [line.strip() for line in f]
# 只保留前VOCAB_SIZE个高频词
vocab = vocab_all[:VOCAB_SIZE-2]
word_index = {"<PAD>":0, "<UNK>":1}
for idx, word in enumerate(vocab):
    word_index[word] = idx + 2

def texts_to_sequences(texts, word_dict, max_len):
    sequences = []
    for text in texts:
        seq = [word_dict.get(word, 1) for word in text.split()]
        seq = seq[:max_len]
        sequences.append(seq)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

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

# ===================== 【修复完成】带Mask支持、类型兼容、输出形状声明的Attention层 =====================
class MaskedAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.att_w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
            name="att_weight"
        )
        super().build(input_shape)

    # 新增：声明输出维度，解决keras形状推断报错
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, inputs, mask=None):
        # 计算原始得分
        score = tf.matmul(tf.reshape(inputs, (-1, inputs.shape[-1])), self.att_w)
        score = tf.reshape(score, (-1, inputs.shape[1]))
        # 统一转换为和score相同浮点类型，消除类型不匹配报错
        if mask is not None:
            mask_float = tf.cast(mask, dtype=score.dtype)
            score = score - (1.0 - mask_float) * 1e9
        attn_weight = tf.nn.softmax(score)
        attn_weight = tf.expand_dims(attn_weight, axis=-1)
        feat = inputs * attn_weight
        return tf.reduce_sum(feat, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

# ===================== 学习率调度：预热 + 余弦退火 =====================
def lr_schedule(epoch):
    if epoch < WARMUP_EPOCHS:
        lr = BASE_LR * (epoch + 1) / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * progress))
    return lr

# ===================== 训练回调函数 =====================
# 改用keras标准格式存储模型，消除h5警告
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    mode="max",
    restore_best_weights=False,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.7,
    patience=4,
    min_lr=MIN_LR,
    verbose=1
)
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
callbacks = [checkpoint, early_stop, reduce_lr, lr_scheduler]

# 优化器：更强梯度裁剪
opt = tf.keras.optimizers.Adam(
    learning_rate=BASE_LR,
    clipnorm=1.5,
    amsgrad=True
)

# ===================== 三层双向LSTM + Mask注意力 + 双分支融合 模型 =====================
print("\n========== 三层BiLSTM+Mask兼容注意力+全局池化融合模型 ==========")
inputs = layers.Input(shape=(MAX_SEQ_LEN,))

# 词嵌入层 + 高斯噪声增强泛化
emb = layers.Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBED_DIM,
    embeddings_regularizer=regularizers.l2(L2_REG),
    mask_zero=True
)(inputs)
emb = layers.GaussianNoise(0.08)(emb)
emb = layers.Dropout(DROPOUT * 0.6)(emb)

# 三层堆叠双向LSTM，逐层加深特征
bi1 = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG))
)(emb)
bi1 = layers.Dropout(DROPOUT)(bi1)

bi2 = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG))
)(bi1)
bi2 = layers.Dropout(DROPOUT)(bi2)

bi3 = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG))
)(bi2)
bi3 = layers.Dropout(DROPOUT)(bi3)

# 分支1：Mask注意力加权特征
att_feat = MaskedAttention()(bi3)
# 分支2：全局最大池化全局特征
pool_feat = layers.GlobalMaxPooling1D()(bi3)
# 特征拼接融合
concat_feat = layers.concatenate([att_feat, pool_feat])

# 多层全连接，阶梯降维
fc1 = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(concat_feat)
fc1 = layers.Dropout(DROPOUT)(fc1)
fc2 = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2_REG))(fc1)
fc2 = layers.Dropout(DROPOUT)(fc2)

# 输出分类
outputs = layers.Dense(num_classes, activation="softmax")(fc2)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=["accuracy"]
)
model.summary()

# 训练
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    shuffle=True
)

# 加载最优模型权重再测试
best_model = tf.keras.models.load_model("best_model.keras", custom_objects={"MaskedAttention": MaskedAttention})
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"\n【最优模型最终结果】测试集准确率：{test_acc:.4f}")

# ===================== 绘制训练曲线 =====================
def plot_curve(hist):
    epochs = range(1, len(hist.history["accuracy"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.history["accuracy"], "c-", linewidth=2, label="训练准确率")
    plt.plot(epochs, hist.history["val_accuracy"], "orange", linestyle="--", linewidth=2, label="验证准确率")
    plt.title("准确率变化曲线")
    plt.xlabel("迭代轮次")
    plt.ylabel("准确率")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.history["loss"], "c-", linewidth=2, label="训练损失")
    plt.plot(epochs, hist.history["val_loss"], "orange", linestyle="--", linewidth=2, label="验证损失")
    plt.title("损失变化曲线")
    plt.xlabel("迭代轮次")
    plt.ylabel("损失值")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_curve(history)

print("\n===== 最优模型汇总 =====")
print(f"测试准确率: {test_acc:.4f}")