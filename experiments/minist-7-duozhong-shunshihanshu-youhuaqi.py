import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ===================== 图像 & 切片可视化 =====================
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
sample_img = x_train[1000]
plt.imshow(sample_img, cmap='gray')
plt.title(f"原始样本 - 数字{y_train[1000]}")
plt.axis('off')

plt.subplot(1, 3, 2)
patch = sample_img[8:20, 8:20]
plt.imshow(patch, cmap='gray')
plt.title("图像中心切片")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sample_img.reshape(784, 1), cmap='gray', aspect='auto')
plt.title("展平为 784 维向量")
plt.axis('off')

plt.suptitle("原始图像 → 切片 → 展平全过程", fontsize=14)
plt.tight_layout()
plt.show()

# ======================== 数据预处理 ========================
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

val_size = 6000
x_val = x_train[-val_size:]
y_val = y_train[-val_size:]
x_train = x_train[:-val_size]
y_train = y_train[:-val_size]

# ======================== 图像增强 ========================
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(x_train)


# ===================== 图像增强对比图 =====================
def show_augmented_img(img, datagen, n=6):
    img = img.reshape(1, 28, 28, 1)
    plt.figure(figsize=(12, 3))
    plt.subplot(1, n + 1, 1)
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title("原始图像")
    plt.axis('off')
    i = 2
    for batch in datagen.flow(img, batch_size=1):
        plt.subplot(1, n + 1, i)
        plt.imshow(batch[0, :, :, 0], cmap='gray')
        plt.title(f"增强{i - 1}")
        plt.axis('off')
        i += 1
        if i > n + 1: break
    plt.suptitle("图像增强效果对比", fontsize=14)
    plt.show()


show_augmented_img(x_train[1000], datagen)


# ===================== 【核心】只对比 3 组组合 =====================
def create_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)), Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)), Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)), Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    return model

configs = [
    {"opt": "adam", "loss": "categorical_crossentropy", "name": "Adam + 交叉熵"},
    {"opt": "sgd", "loss": "categorical_crossentropy", "name": "SGD + 交叉熵"},
    {"opt": "adam", "loss": "mse", "name": "Adam + MSE"},
]

results = []
histories = []

# 开始训练对比
for cfg in configs:
    print("\n========================================")
    print(f"正在训练：{cfg['name']}")
    print("========================================")

    model = create_model()
    model.compile(optimizer=cfg["opt"], loss=cfg["loss"], metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results.append({"name": cfg["name"], "acc": test_acc})
    histories.append(history)

# ======================== 输出对比表格 ========================
print("\n" + "=" * 50)
print("优化器 & 损失函数 最终测试准确率对比")
print("=" * 50)
for r in results:
    print(f"{r['name']:20s} | 准确率 = {r['acc']:.4f}")

# ======================== 绘制对比曲线 ========================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for i, h in enumerate(histories):
    plt.plot(h.history['val_accuracy'], label=configs[i]['name'])
plt.title("验证集准确率对比")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
for i, h in enumerate(histories):
    plt.plot(h.history['val_loss'], label=configs[i]['name'])
plt.title("验证集损失对比")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()