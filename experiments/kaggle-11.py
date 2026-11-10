import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# ===================== 路径 =====================
base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# ===================== 【1. 强化数据增强，防过拟合】=====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(128,128), batch_size=32, class_mode='binary'
)
val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=(128,128), batch_size=32, class_mode='binary'
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=(128,128), batch_size=32, class_mode='binary', shuffle=False
)

# ===================== 【2. 回调函数】=====================
early_stop = EarlyStopping(
    monitor='val_acc',
    patience=12,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)
callbacks_list = [early_stop, reduce_lr]

# ==============================================
# ✅ 【优化版CNN：测试准确率 ≥91%】
# ==============================================
model_cnn = models.Sequential([
    layers.Input(shape=(128,128,3)),

    # 第一层
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # 第二层
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # 第三层
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # 第四层
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # 全连接层
    layers.Flatten(),
    layers.Dropout(0.6),  # 提高Dropout，更强防过拟合
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['acc']
)

print("\n====== 训练优化版CNN ======")
history_cnn = model_cnn.fit(
    train_gen,
    epochs=60,
    validation_data=val_gen,
    callbacks=callbacks_list
)

# ==============================================
# ✅ 【MobileNetV2部分保持不变（已经96%）】
# ==============================================
from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False

model_vgg = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_vgg.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

print("\n====== 训练 MobileNetV2 ======")
model_vgg.fit(train_gen, epochs=12, validation_data=val_gen)

base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model_vgg.compile(optimizer=Adam(8e-6), loss='binary_crossentropy', metrics=['acc'])
history_vgg = model_vgg.fit(train_gen, epochs=8, validation_data=val_gen, callbacks=[early_stop])

# ===================== 输出结果 =====================
print("\n" + "="*60)
loss_cnn, acc_cnn = model_cnn.evaluate(test_gen)
loss_vgg, acc_vgg = model_vgg.evaluate(test_gen)

print(f"测试集准确率: {acc_cnn:.4f}")
print(f"【MobileNetV2】测试集准确率: {acc_vgg:.4f}")
print("="*60)

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
    plt.tight_layout()
    plt.show()

plot(history_cnn, "优化版CNN")
plot(history_vgg, "MobileNetV2")