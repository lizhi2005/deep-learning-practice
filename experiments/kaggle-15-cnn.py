import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ---------------------- 关键：修改为你自己的目录路径 ----------------------
BASE_PATH = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"

train_dir = os.path.join(BASE_PATH, "train")
val_dir = os.path.join(BASE_PATH, "validation")
test_dir = os.path.join(BASE_PATH, "test")

# 超参数
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# 1. 训练集：数据增强 + 归一化
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# 2. 验证/测试集：仅归一化，不增强
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 3. 生成数据流
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"  # 二分类转为2维独热编码
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # 测试集不打乱，方便结果对照
)

# 通用回调函数（早停+学习率衰减）
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

def build_deep_cnn():
    model = models.Sequential([
        # 第1卷积块
        layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=INPUT_SHAPE),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # 第2卷积块
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # 第3卷积块
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # 第4卷积块
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # 第5卷积块
        layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # 全连接层
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # 输出层：2分类 + Softmax
        layers.Dense(2, activation="softmax")
    ])

    # 模型编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

# 构建并训练模型
cnn_model = build_deep_cnn()
cnn_model.summary()

print("===== 开始训练深度CNN模型 =====")
history_cnn = cnn_model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks
)

# 绘制训练曲线
def plot_history(history, title_name):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, "r", label="Train Acc")
    plt.plot(epochs, val_acc, "b", label="Val Acc")
    plt.title(f"{title_name} Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, "r", label="Train Loss")
    plt.plot(epochs, val_loss, "b", label="Val Loss")
    plt.title(f"{title_name} Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history_cnn, "Deep CNN")

# 验证集评估
val_loss_cnn, val_acc_cnn = cnn_model.evaluate(val_generator)
print(f"深度CNN 验证集准确率: {val_acc_cnn:.4f}")

# 测试集分类预测
test_loss_cnn, test_acc_cnn = cnn_model.evaluate(test_generator)
print(f"深度CNN 测试集准确率: {test_acc_cnn:.4f}")

# 保存模型
cnn_model.save("cats_dogs_deep_cnn.keras")
print("模型已保存为 cats_dogs_deep_cnn.keras")







