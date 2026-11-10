import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# ---------------------- 1. 加载数据 ----------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 归一化 & 增加通道维度
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)  # (60000,28,28,1)
x_test = np.expand_dims(x_test, axis=-1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ---------------------- 2. 绘制原始图像 ----------------------
def plot_raw_images(images, labels, num=5):
    plt.figure(figsize=(10,2))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Label:{np.argmax(labels[i])}")
        plt.axis("off")
    plt.suptitle("Raw MNIST Images", y=1.02)
    plt.show()

plot_raw_images(x_train, y_train)

# ---------------------- 3. 图像切片（中心裁剪） ----------------------
def plot_sliced_image(img):
    # 原始 28×28
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title("Original (28×28)")
    plt.axis("off")

    # 中心切片：8:20 → 12×12
    patch = img[8:20, 8:20, :]
    plt.subplot(1,2,2)
    plt.imshow(patch.squeeze(), cmap="gray")
    plt.title("Sliced (12×12)")
    plt.axis("off")
    plt.show()

# 取第一张演示
plot_sliced_image(x_train[0])

# ---------------------- 4. 图像增强定义 & 展示 ----------------------
datagen = ImageDataGenerator(
    rotation_range=10,      # 随机旋转 ±10°
    width_shift_range=0.1,  # 水平平移 ±10%
    height_shift_range=0.1, # 垂直平移 ±10%
    zoom_range=0.1,         # 缩放 0.9–1.1
    fill_mode="constant",
    cval=0.0
)

def plot_augmented_samples(img, n=5):
    plt.figure(figsize=(12,2.5))
    plt.subplot(1, n+1, 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    img = np.expand_dims(img, axis=0)  # (1,28,28,1)
    for i in range(n):
        aug_img = datagen.random_transform(img[0])
        plt.subplot(1, n+1, i+2)
        plt.imshow(aug_img.squeeze(), cmap="gray")
        plt.title(f"Aug {i+1}")
        plt.axis("off")
    plt.suptitle("Image Augmentation Examples", y=1.02)
    plt.show()

plot_augmented_samples(x_train[0])

# ---------------------- 5. 构建相同的全连接模型 ----------------------
def build_model():
    model = Sequential([
        Flatten(input_shape=(28,28,1)),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ---------------------- 6. 训练：无增强 ----------------------
print("=== Training WITHOUT Augmentation ===")
model_no_aug = build_model()
history_no_aug = model_no_aug.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ---------------------- 7. 训练：有增强 ----------------------
print("\n=== Training WITH Augmentation ===")
model_aug = build_model()
history_aug = model_aug.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=15,
    validation_data=(x_test, y_test),
    verbose=1
)

# ---------------------- 8. 对比测试集结果 ----------------------
def evaluate(model, name):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} → Test Acc: {acc:.4f}, Test Loss: {loss:.4f}")

evaluate(model_no_aug, "Without Aug")
evaluate(model_aug, "With Aug")

# ---------------------- 9. 绘制训练曲线对比 ----------------------
def plot_history(hist_no, hist_aug):
    plt.figure(figsize=(12,4))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(hist_no.history["accuracy"], label="NoAug-Train")
    plt.plot(hist_no.history["val_accuracy"], label="NoAug-Val")
    plt.plot(hist_aug.history["accuracy"], label="Aug-Train")
    plt.plot(hist_aug.history["val_accuracy"], label="Aug-Val")
    plt.title("Accuracy Curve")
    plt.legend()
    # Loss
    plt.subplot(1,2,2)
    plt.plot(hist_no.history["loss"], label="NoAug-Train")
    plt.plot(hist_no.history["val_loss"], label="NoAug-Val")
    plt.plot(hist_aug.history["loss"], label="Aug-Train")
    plt.plot(hist_aug.history["val_loss"], label="Aug-Val")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

plot_history(history_no_aug, history_aug)