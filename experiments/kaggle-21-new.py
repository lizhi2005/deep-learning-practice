# 放在最顶部，关闭 oneDNN 警告
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽冗余tf日志

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

# ===================== 2. 适度数据增强可视化（缩小扰动范围，减少噪声） =====================
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
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

# ===================== 3. 搭建优化Deep CNN（下调正则强度，解决欠拟合） =====================
reg = regularizers.l2(1e-4)  # 由1e-3下调，减轻权重惩罚
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
    # 全局平均池化替代Flatten，大幅减少参数量，防过拟合
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),  # 降低dropout，避免过度抑制特征学习
    layers.Dense(128, activation='relu', kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# 编译：调高学习率至1e-4，加速CPU收敛
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=['acc']
)

# ===================== 4. 构建数据生成器 =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    brightness_range=[0.75, 1.25]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# CPU适配batch=16，内存占用低
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

# ===================== 5. 训练回调（缩短早停等待，节约CPU算力） =====================
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
best_ckpt = callbacks.ModelCheckpoint("best_deep_cnn.keras", monitor="val_acc", save_best_only=True, verbose=1)
lr_decay = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=6, min_lr=1e-7, verbose=1)
callback_list = [early_stop, best_ckpt, lr_decay]

# ===================== 6. 模型训练 =====================
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

# ===================== 7. 绘制训练曲线 =====================
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
plt.show()

# ===================== 8. 加载最优模型做测试集评估 =====================
print("\n=== 加载全局最优模型进行测试集评估 ===")
best_model = models.load_model("best_deep_cnn.keras")
test_loss, test_acc = best_model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")