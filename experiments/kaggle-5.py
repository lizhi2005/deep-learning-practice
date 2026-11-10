import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径配置
original_train_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\kaggle-train"
base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# 【优化：缩小增强幅度】
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
valtest_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode="binary")
val_gen = valtest_datagen.flow_from_directory(val_dir, target_size=(150,150), batch_size=32, class_mode="binary")
test_gen = valtest_datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=32, class_mode="binary")

# 【优化早停：patience=7】
early_stop = EarlyStopping(monitor="val_acc", patience=7, restore_best_weights=True)

# ==========优化后CNN结构==========
model_cnn = models.Sequential([
    layers.Input(shape=(150,150,3)),
    layers.Conv2D(32,(3,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dropout(0.45),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'), # 新增全连接层
    layers.Dense(1, activation='sigmoid')
])
model_cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
history_cnn = model_cnn.fit(train_gen, epochs=45, validation_data=val_gen, callbacks=[early_stop])

# ==========优化VGG16微调==========
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.trainable = False

model_vgg = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])
# 冻结主干训练15轮
model_vgg.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['acc'])
model_vgg.fit(train_gen, epochs=15, validation_data=val_gen)

# 微调：前16层冻结，放开末尾卷积
conv_base.trainable = True
freeze_num = 16
for layer in conv_base.layers[:freeze_num]:
    layer.trainable = False
# 极小学习率微调12轮
model_vgg.compile(optimizer=Adam(learning_rate=8e-6), loss='binary_crossentropy', metrics=['acc'])
history_vgg = model_vgg.fit(train_gen, epochs=12, validation_data=val_gen, callbacks=[early_stop])

# 测试
print("="*60)
cnn_loss, cnn_acc = model_cnn.evaluate(test_gen)
print(f"优化后CNN测试准确率:{cnn_acc:.4f}")
print("="*60)
vgg_loss, vgg_acc = model_vgg.evaluate(test_gen)
print(f"优化后VGG16测试准确率:{vgg_acc:.4f}")

# 绘图
def plot(history, name):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    ep = range(len(acc))
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(ep,acc,label='训练准确率')
    plt.plot(ep,val_acc,label='验证准确率')
    plt.title(f"{name}准确率")
    plt.legend()
    plt.subplot(122)
    plt.plot(ep,loss,label='训练损失')
    plt.plot(ep,val_loss,label='验证损失')
    plt.title(f"{name}损失")
    plt.legend()
    plt.show()
plot(history_cnn,"深度CNN")
plot(history_vgg,"VGG16迁移学习")