import os
import shutil
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

original_train_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\kaggle-train"
base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# ===================== 自动创建文件夹 =====================
def make_dirs():
    for root in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(root, "cat"), exist_ok=True)
        os.makedirs(os.path.join(root, "dog"), exist_ok=True)
make_dirs()

# ===================== 自动复制图片（已打开！） =====================
def copy_data(start, end, dst_root):
    cat_dst = os.path.join(dst_root, "cat")
    dog_dst = os.path.join(dst_root, "dog")
    for i in range(start, end):
        shutil.copy(os.path.join(original_train_dir, f"cat.{i}.jpg"), cat_dst)
        shutil.copy(os.path.join(original_train_dir, f"dog.{i}.jpg"), dog_dst)

copy_data(0, 1000, train_dir)
copy_data(1000, 1500, val_dir)
copy_data(1500, 2000, test_dir)

# ===================== 数据生成器 =====================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
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

# ===================== 模型1：深度CNN =====================
from tensorflow.keras import layers, models
model_cnn = models.Sequential([
    layers.Input(shape=(150,150,3)),
    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512,activation="relu"),
    layers.Dense(1,activation="sigmoid")
])

model_cnn.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history_cnn = model_cnn.fit(train_gen, epochs=35, validation_data=val_gen)

# ===================== 模型2：VGG16 =====================
from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
conv_base.trainable = False

model_vgg = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256,activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1,activation="sigmoid")
])

model_vgg.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
history_vgg = model_vgg.fit(train_gen, epochs=18, validation_data=val_gen)

# ===================== 测试集评估 =====================
print("="*50)
print("CNN 测试集结果：")
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

plot(history_cnn, "深度CNN")
plot(history_vgg, "VGG16")