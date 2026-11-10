# 放在最顶部，关闭 oneDNN 警告
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# ===================== 1. 统一配置路径 =====================
BASE_PATH = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(BASE_PATH, "train")
validation_dir = os.path.join(BASE_PATH, "validation")
test_dir = os.path.join(BASE_PATH, "test")

train_cats_dir = os.path.join(train_dir, "cat")

# 路径&数据数量检查
print("=== 数据集路径检查 ===")
print("数据集根目录:", BASE_PATH)
print("训练集目录:", train_dir)
print("训练集猫图片数量:", len(os.listdir(train_cats_dir)))
print("训练集狗图片数量:", len(os.listdir(os.path.join(train_dir, "dog"))))
print("验证集猫图片数量:", len(os.listdir(os.path.join(validation_dir, "cat"))))
print("验证集狗图片数量:", len(os.listdir(os.path.join(validation_dir, "dog"))))
print("测试集猫图片数量:", len(os.listdir(os.path.join(test_dir, "cat"))))
print("测试集狗图片数量:", len(os.listdir(os.path.join(test_dir, "dog"))))

# ===================== 2. 数据增强可视化 =====================
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[0]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

plt.figure(figsize=(10, 4))
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(1, 4, i+1)
    plt.imshow(image.array_to_img(batch[0]))
    plt.axis('off')
    i += 1
    if i % 4 == 0:
        break
plt.suptitle("Data Augmentation Examples")
plt.show()

# ===================== 3. 【优化】加深加宽CNN模型 =====================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # 提升通道数至256，增大模型容量
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    # 增加Dropout，更强防过拟合
    layers.Dropout(0.5),
    # 增大全连接层节点
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# 【优化】更换优化器+调小学习率，收敛更稳定
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=5e-5),
    metrics=['acc']
)

# ===================== 4. 【优化】调整批次大小 =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# batch_size 从32改为64
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    shuffle=False
)

# ===================== 5. 【优化】增加训练轮数 =====================
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=120,          # 从100轮提升到120轮，充分收敛
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# 【修复】使用新版keras格式保存模型，消除警告
model.save('cats_and_dogs_small_2.keras')

# ===================== 6. 【优化】平滑曲线图（适合报告） =====================
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.rcParams['font.sans-serif'] = ['SimHei']   # 支持中文
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 5))
# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r-', linewidth=1.5, label='训练集准确率')
plt.plot(epochs, val_acc, 'b-', linewidth=1.5, label='验证集准确率')
plt.title('训练集 & 验证集 准确率变化')
plt.xlabel('训练轮数 Epochs')
plt.ylabel('准确率 Accuracy')
plt.legend()
plt.grid(alpha=0.3)

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r-', linewidth=1.5, label='训练集损失')
plt.plot(epochs, val_loss, 'b-', linewidth=1.5, label='验证集损失')
plt.title('训练集 & 验证集 损失变化')
plt.xlabel('训练轮数 Epochs')
plt.ylabel('损失 Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===================== 7. 测试集评估 =====================
print("\n=== 测试集评估 ===")
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")