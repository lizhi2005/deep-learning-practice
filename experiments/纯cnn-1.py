import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")
# 数据增强与生成器
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
    train_dir, target_size=(150,150), batch_size=32, class_mode='binary'
)
val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=(150,150), batch_size=32, class_mode='binary'
)
test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=(150,150), batch_size=32, class_mode='binary', shuffle=False
)
# CNN回调函数
early_stop_cnn = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cnn = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
callbacks_cnn = [early_stop_cnn, reduce_lr_cnn]
#四层双卷积深度CNN模型
model_cnn = models.Sequential([
    layers.Input(shape=(150,150,3)),
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    # Block 3
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    # Block 4
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    # 分类头
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
# 模型编译
model_cnn.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['acc']
)
print("\n========== 开始训练深度CNN模型 ==========")
history_cnn = model_cnn.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    callbacks=callbacks_cnn
)
# 测试集评估
print("\n" + "="*60)
test_loss, test_acc = model_cnn.evaluate(test_gen)
print(f"【深度CNN】测试集准确率: {test_acc:.4f}")
print(f"【深度CNN】测试集损失: {test_loss:.4f}")
print("="*60)
# 绘制训练曲线
def plot_train_curve(history, title):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(epochs, acc, label='训练准确率')
    plt.plot(epochs, val_acc, label='验证准确率')
    plt.title(title + ' 准确率曲线')
    plt.legend()
    plt.subplot(122)
    plt.plot(epochs, loss, label='训练损失')
    plt.plot(epochs, val_loss, label='验证损失')
    plt.title(title + ' 损失曲线')
    plt.tight_layout()
    plt.show()
plot_train_curve(history_cnn, "深度CNN")