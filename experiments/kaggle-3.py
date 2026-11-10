import os
import shutil
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 你的路径 =====================
original_train_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\kaggle-train"
base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# ===================== 数据增强（增强版） =====================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,        # 增强
    width_shift_range=0.3,   # 增强
    height_shift_range=0.3,  # 增强
    shear_range=0.3,         # 增强
    zoom_range=0.3,          # 增强
    horizontal_flip=True,
    fill_mode='nearest'
)

valtest_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(150,150), batch_size=32, class_mode="binary"
)
val_gen = valtest_datagen.flow_from_directory(
    val_dir, target_size=(150,150), batch_size=32, class_mode="binary"
)
test_gen = valtest_datagen.flow_from_directory(
    test_dir, target_size=(150,150), batch_size=32, class_mode="binary"
)

# ===================== 🔥 优化后的深度 CNN（达标90%+） =====================
from tensorflow.keras import layers, models

model_cnn = models.Sequential([
    layers.Input(shape=(150,150,3)),

    layers.Conv2D(32,(3,3)),
    layers.BatchNormalization(),  # 新增
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3)),
    layers.BatchNormalization(),  # 新增
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3)),
    layers.BatchNormalization(),  # 新增
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3)),
    layers.BatchNormalization(),  # 新增
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dropout(0.4),  # 改成 0.4
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 优化器改成 Adam
model_cnn.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["acc"])

# 训练 45 轮
history_cnn = model_cnn.fit(train_gen,
                            epochs=45,
                            validation_data=val_gen)

# ===================== VGG16 迁移学习（95%+） =====================
from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
conv_base.trainable = False

model_vgg = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_vgg.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["acc"])

history_vgg = model_vgg.fit(train_gen,
                            epochs=20,
                            validation_data=val_gen)

# ===================== 测试集评估 =====================
print("="*50)
print("优化后 CNN 测试集结果：")
loss_cnn, acc_cnn = model_cnn.evaluate(test_gen)
print(f"准确率：{acc_cnn:.4f}")

print("="*50)
print("VGG16 测试集结果：")
loss_vgg, acc_vgg = model_vgg.evaluate(test_gen)
print(f"准确率：{acc_vgg:.4f}")

# ===================== 绘图 =====================
def plot(history, name):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label="训练准确率")
    plt.plot(epochs, val_acc, label="验证准确率")
    plt.title(f"{name} 准确率")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label="训练损失")
    plt.plot(epochs, val_loss, label="验证损失")
    plt.title(f"{name} 损失")
    plt.legend()
    plt.show()

plot(history_cnn, "优化后深度CNN")
plot(history_vgg, "VGG16")