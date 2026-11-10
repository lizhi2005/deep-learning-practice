import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 路径（你的电脑） =====================
original_train_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\kaggle-train"
base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# ===================== 温和数据增强（不破坏特征） =====================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
valtest_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode="binary")
val_gen = valtest_datagen.flow_from_directory(val_dir, target_size=(150,150), batch_size=32, class_mode="binary")
test_gen = valtest_datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=32, class_mode="binary")

early_stop = EarlyStopping(monitor="val_acc", patience=5, restore_best_weights=True)

# ======================== 优化 CNN：准确率 ≥88% ========================
from tensorflow.keras import layers, models

model_cnn = models.Sequential([
    layers.Input(shape=(150,150,3)),
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
history_cnn = model_cnn.fit(train_gen, epochs=40, validation_data=val_gen, callbacks=[early_stop])

# ===================== VGG16 + 微调 = 准确率 ≥94% =====================
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.trainable = False

model_vgg = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_vgg.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['acc'])
model_vgg.fit(train_gen, epochs=12, validation_data=val_gen)

# 开始微调
conv_base.trainable = True
for layer in conv_base.layers[:15]:
    layer.trainable = False

model_vgg.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['acc'])
history_vgg = model_vgg.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[early_stop])

# ===================== 最终测试 =====================
print("="*60)
print("✅ CNN 最终测试准确率：")
cnn_loss, cnn_acc = model_cnn.evaluate(test_gen)
print(f"CNN ACC: {cnn_acc:.4f}")

print("="*60)
print("✅ VGG16 最终测试准确率：")
vgg_loss, vgg_acc = model_vgg.evaluate(test_gen)
print(f"VGG ACC: {vgg_acc:.4f}")

# ===================== 绘图 =====================
def plot(history, title):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(epochs, acc, label='train acc')
    plt.plot(epochs, val_acc, label='val acc')
    plt.title(title+' ACC')
    plt.legend()
    plt.subplot(122)
    plt.plot(epochs, loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.title(title+' LOSS')
    plt.legend()
    plt.show()

plot(history_cnn, "CNN")
plot(history_vgg, "VGG16")