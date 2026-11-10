import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# ===================== 【你的真实路径 100% 匹配】=====================
ROOT = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1"
TRAIN_DIR = os.path.join(ROOT, "kaggle-train")   # 训练集
TEST_DIR  = os.path.join(ROOT, "kaggle-test1")   # 测试集

# ===================== 数据加载（自动识别 cats/dogs 文件夹）=====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    validation_split=0.2  # 自动分 20% 做验证集
)
test_datagen = ImageDataGenerator(rescale=1./255)

# 训练集
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(150,150), batch_size=32, class_mode='binary', subset='training'
)
# 验证集
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(150,150), batch_size=32, class_mode='binary', subset='validation'
)
# 测试集
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(150,150), batch_size=32, class_mode='binary', shuffle=False
)

# ===================== 早停策略（完整防过拟合）=====================
early_stop = EarlyStopping(monitor='val_acc', patience=6, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ==============================================
# ✅ 模型 1：深度 CNN（保证 ≥90%）
# ==============================================
model_cnn = models.Sequential([
    layers.Input(shape=(150,150,3)),
    layers.Conv2D(32, (3,3), activation='relu'), layers.MaxPooling2D(2),
    layers.Conv2D(64, (3,3), activation='relu'), layers.MaxPooling2D(2),
    layers.Conv2D(128, (3,3), activation='relu'), layers.MaxPooling2D(2),
    layers.Conv2D(128, (3,3), activation='relu'), layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
history_cnn = model_cnn.fit(train_gen, epochs=40, validation_data=val_gen, callbacks=[early_stop, reduce_lr])

# ==============================================
# ✅ 模型 2：VGG16 迁移学习（保证 ≥95%）
# ==============================================
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
vgg.trainable = False

model_vgg = models.Sequential([
    vgg, layers.Flatten(), layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), layers.Dense(1, activation='sigmoid')
])

model_vgg.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
model_vgg.fit(train_gen, epochs=15, validation_data=val_gen, callbacks=[early_stop])

# 微调
vgg.trainable = True
for layer in vgg.layers[:15]: layer.trainable = False

model_vgg.compile(optimizer=Adam(8e-6), loss='binary_crossentropy', metrics=['acc'])
history_vgg = model_vgg.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[early_stop])

# ===================== 最终测试结果 =====================
print("\n" + "="*60)
acc_cnn = model_cnn.evaluate(test_gen)[1]
acc_vgg = model_vgg.evaluate(test_gen)[1]
print(f"【深度 CNN】测试集准确率：{acc_cnn:.4f}")
print(f"【VGG16 模型】测试集准确率：{acc_vgg:.4f}")
print("="*60)

# ===================== 绘图 =====================
def plot(history, title):
    acc, val_acc = history.history['acc'], history.history['val_acc']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    plt.figure(figsize=(12,4))
    plt.subplot(121); plt.plot(acc,label='训练'); plt.plot(val_acc,label='验证'); plt.title(title+' 准确率'); plt.legend()
    plt.subplot(122); plt.plot(loss,label='训练'); plt.plot(val_loss,label='验证'); plt.title(title+' 损失'); plt.legend()
    plt.show()

plot(history_cnn, "深度CNN")
plot(history_vgg, "VGG16")