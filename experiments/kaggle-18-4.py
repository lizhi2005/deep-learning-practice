# 放在最顶部，关闭 oneDNN 警告
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# ===================== 1. 统一配置正确路径 =====================
BASE_PATH = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(BASE_PATH, "train")
validation_dir = os.path.join(BASE_PATH, "validation")
test_dir = os.path.join(BASE_PATH, "test")

# 训练集猫文件夹路径
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

# ===================== 2. 弱化数据增强，减少噪声干扰 =====================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=[0.9, 1.1],
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# 读取一张猫的图片做增强演示
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

# ===================== 3. 模型结构（保持不变，仅微调正则强度） =====================
reg = regularizers.l2(1e-4)
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    # 卷积块1
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # 卷积块2
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # 卷积块3
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # 卷积块4
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # 全局平均池化
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# ===================== 4. 关键修改：降低学习率，适配CPU小样本 =====================
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=5e-5),
    metrics=['acc']
)

# ===================== 5. 构建数据生成器 =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=[0.9, 1.1],
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# ===================== 6. 优化回调：更小衰减系数、缩短早停等待 =====================
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
best_ckpt = callbacks.ModelCheckpoint("best_cnn.keras", monitor="val_acc", save_best_only=True, verbose=1)
lr_decay = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, min_lr=1e-7, verbose=1)
callback_list = [early_stop, best_ckpt, lr_decay]

# ===================== 7. 模型训练 =====================
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callback_list
)

model.save('cats_and_dogs_small_final.keras')

# ===================== 8. 绘制训练曲线 =====================
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.tight_layout()
plt.show()

# ===================== 9. 加载最优模型做测试集评估 =====================
print("\n=== 加载全局最优模型进行测试集评估 ===")
best_model = models.load_model("best_cnn.keras")
test_loss, test_acc = best_model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")