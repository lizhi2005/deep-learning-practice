import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 你的 Windows 路径（直接用！） =====================
base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# ===================== 数据增强（温和、不破坏图片、最稳！） =====================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
valtest_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(150,150), batch_size=32, class_mode='binary'
)
val_gen = valtest_datagen.flow_from_directory(
    val_dir, target_size=(150,150), batch_size=32, class_mode='binary'
)
test_gen = valtest_datagen.flow_from_directory(
    test_dir, target_size=(150,150), batch_size=32, class_mode='binary'
)

# 自动保存最优模型 + 学习率自动下降
early_stop = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# ===================== ✅ 你最初的成功 CNN 结构 + 优化（必上90%） =====================
model_cnn = models.Sequential([
    layers.Input(shape=(150,150,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history_cnn = model_cnn.fit(
    train_gen,
    epochs=40,
    validation_data=val_gen,
    callbacks=[early_stop, reduce_lr]
)

# ===================== 🚀 VGG16 微调（95%+） =====================
from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.trainable = False

model_vgg = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_vgg.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])
model_vgg.fit(train_gen, epochs=15, validation_data=val_gen)

# 解冻微调
conv_base.trainable = True
for layer in conv_base.layers[:15]:
    layer.trainable = False

model_vgg.compile(optimizer=Adam(learning_rate=8e-6),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history_vgg = model_vgg.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[early_stop]
)

# ===================== 最终测试 =====================
print("="*60)
loss_cnn, acc_cnn = model_cnn.evaluate(test_gen)
print(f"✅ CNN 测试准确率: {acc_cnn:.4f}")

print("="*60)
loss_vgg, acc_vgg = model_vgg.evaluate(test_gen)
print(f"✅ VGG16 测试准确率: {acc_vgg:.4f}")

# ===================== 绘图 =====================
def plot(history, title):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(epochs, acc, label='训练准确率')
    plt.plot(epochs, val_acc, label='验证准确率')
    plt.title(title + ' 准确率')
    plt.legend()

    plt.subplot(122)
    plt.plot(epochs, loss, label='训练损失')
    plt.plot(epochs, val_loss, label='验证损失')
    plt.title(title + ' 损失')
    plt.legend()
    plt.show()

plot(history_cnn, "深度CNN")
plot(history_vgg, "VGG16")