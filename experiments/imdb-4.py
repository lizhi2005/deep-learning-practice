# ===================== IMDB 三种优化器对比实验 =====================
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭烦人的oneDNN警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 关闭TF多余日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ===================== 超参数 =====================
max_words = 10000
epochs = 20
batch_size = 256

# ===================== 加载数据 =====================
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)


# ===================== 文本向量化 =====================
def vectorize_sequences(sequences, dimension=max_words):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.0
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.array(train_labels).astype("float32")
y_test = np.array(test_labels).astype("float32")

# 验证集
x_val = x_train[:5000]
y_val = y_train[:5000]
x_train_part = x_train[5000:]
y_train_part = y_train[5000:]


# ===================== 模型构建函数 =====================
def build_model():
    model = Sequential([
        Input(shape=(max_words,)),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
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
    print("\n==================================")
    print(f"正在训练：{cfg['name']}")
    print("==================================")

    model = build_model()
    model.compile(
        optimizer=cfg["opt"],
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    h = model.fit(
        x_train_part, y_train_part,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    results.append({
        "optimizer": cfg["name"],
        "test_acc": test_acc,
        "test_loss": test_loss
    })
    histories.append(h)

# ===================== 输出三种优化器对比结果 =====================
print("\n" + "=" * 50)
print("三种优化器最终测试结果对比")
print("=" * 50)
for r in results:
    print(f"{r['optimizer']:10s} | 准确率：{r['test_acc']:.4f} | 损失：{r['test_loss']:.4f}")

# ===================== 绘制对比曲线 =====================
plt.figure(figsize=(12, 5))

# 准确率
plt.subplot(1, 2, 1)
for i, name in enumerate(["Adam", "SGD", "RMSprop"]):
    plt.plot(histories[i].history["val_accuracy"], label=name)
plt.title("三种优化器 验证准确率对比")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend()

# 损失
plt.subplot(1, 2, 2)
for i, name in enumerate(["Adam", "SGD", "RMSprop"]):
    plt.plot(histories[i].history["val_loss"], label=name)
plt.title("三种优化器 验证损失对比")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()