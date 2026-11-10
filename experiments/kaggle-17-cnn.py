import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ========== 路径（保持你原有目录不变） ==========
BASE_PATH = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(BASE_PATH, "train")
val_dir = os.path.join(BASE_PATH, "validation")
test_dir = os.path.join(BASE_PATH, "test")

# 超参数
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)

# ========== 强化数据增强 ==========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# 数据加载
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
val_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

# ========== 回调函数 ==========
callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1)
]

# ========== 优化后的深度CNN模型 ==========
def build_optimized_cnn():
    inputs = Input(shape=INPUT_SHAPE)
    x = inputs

    # 卷积块1
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # 卷积块2
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # 卷积块3
    x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # 卷积块4
    x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # 卷积块5
    x = layers.Conv2D(512, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # 全连接
    x = layers.Flatten()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 训练
cnn_model = build_optimized_cnn()
cnn_model.summary()

print("===== 开始训练优化版深度CNN =====")
history_cnn = cnn_model.fit(
    train_generator,
    epochs=60,
    validation_data=val_generator,
    callbacks=callbacks
)

# 绘制曲线
def plot_history(history, title):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["loss"]
    epochs = range(len(acc))

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, "r", label="Train Acc")
    plt.plot(epochs, val_acc, "b", label="Val Acc")
    plt.title(f"{title} Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, "r", label="Train Loss")
    plt.plot(epochs, val_loss, "b", label="Val Loss")
    plt.title(f"{title} Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history_cnn, "Optimized Deep CNN")

# 评估
val_loss, val_acc = cnn_model.evaluate(val_generator)
test_loss, test_acc = cnn_model.evaluate(test_generator)
print(f"【深度CNN】验证集准确率: {val_acc:.4f}")
print(f"【深度CNN】测试集准确率: {test_acc:.4f}")

# 保存模型
cnn_model.save("cats_dogs_optimized_cnn.keras")
print("CNN模型已保存")