# ===================== 库导入与环境设置 =====================
import numpy as np
import matplotlib.pyplot as plt
import os

# 关闭TensorFlow多余警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 绘图中文支持
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 导入Keras相关模块
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ===================== 超参数设置 =====================
max_words = 10000  # 保留最高频的10000个词
epochs = 20  # 最大训练轮次
batch_size = 256  # 批次大小

# ===================== 加载IMDB数据集 =====================
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)


# ===================== 文本向量化（BOW词袋模型） =====================
def vectorize_sequences(sequences, dimension=max_words):
    # 创建全零矩阵
    results = np.zeros((len(sequences), dimension))
    # 将对应词位置设为1
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.0
    return results


# 对训练集和测试集进行向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 标签转换为浮点型
y_train = np.array(train_labels).astype("float32")
y_test = np.array(test_labels).astype("float32")

# 划分验证集（前5000条）
x_val = x_train[:5000]
y_val = y_train[:5000]
x_train_part = x_train[5000:]
y_train_part = y_train[5000:]


# ===================== 构建神经网络模型 =====================
def build_model():
    model = Sequential([
        Input(shape=(max_words,)),  # 输入层：10000维向量
        Dense(64, activation="relu"),  # 第一层隐藏层：64个神经元
        Dropout(0.4),  # Dropout随机丢弃40%神经元，防止过拟合
        Dense(32, activation="relu"),  # 第二层隐藏层：32个神经元
        Dropout(0.3),  # Dropout随机丢弃30%神经元
        Dense(1, activation="sigmoid")  # 输出层：1个神经元，二分类概率
    ])
    return model


# ===================== 三种优化器配置 =====================
optimizers = [
    {"name": "Adam", "opt": "adam"},
    {"name": "SGD", "opt": "sgd"},
    {"name": "RMSprop", "opt": "rmsprop"}
]

results = []
histories = []

# ===================== 循环训练三种优化器 =====================
for cfg in optimizers:
    print(f"\n===== 正在训练：{cfg['name']} =====")

    # 构建模型
    model = build_model()

    # 编译模型
    model.compile(
        optimizer=cfg["opt"],
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 早停策略：验证损失连续2轮不下降则停止
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    # 训练模型
    h = model.fit(
        x_train_part, y_train_part,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # 在测试集上评估
    test_loss, test_acc = model.evaluate(x_test, y_test)
    results.append({
        "optimizer": cfg["name"],
        "test_acc": test_acc,
        "test_loss": test_loss
    })
    histories.append(h)

# ===================== 输出对比结果 =====================
print("\n===== 三种优化器测试结果对比 =====")
for r in results:
    print(f"{r['optimizer']:10s} | 准确率：{r['test_acc']:.4f} | 损失：{r['test_loss']:.4f}")

# ===================== 绘制准确率与损失曲线 =====================
plt.figure(figsize=(12, 5))

# 验证准确率对比
plt.subplot(1, 2, 1)
for i, name in enumerate(["Adam", "SGD", "RMSprop"]):
    plt.plot(histories[i].history["val_accuracy"], label=name)
plt.title("三种优化器 验证准确率对比")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# 验证损失对比
plt.subplot(1, 2, 2)
for i, name in enumerate(["Adam", "SGD", "RMSprop"]):
    plt.plot(histories[i].history["val_loss"], label=name)
plt.title("三种优化器 验证损失对比")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()