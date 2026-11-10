# ========== 全局环境配置，屏蔽警告 ==========
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
# 分开导入
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# ========== 1. 数据集路径配置（修改为你本地文件夹路径） ==========
BASE_PATH = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(BASE_PATH, "train")
validation_dir = os.path.join(BASE_PATH, "validation")
test_dir = os.path.join(BASE_PATH, "test")

train_cats_dir = os.path.join(train_dir, "cat")
train_dogs_dir = os.path.join(train_dir, "dog")
val_cats_dir = os.path.join(validation_dir, "cat")
val_dogs_dir = os.path.join(validation_dir, "dog")
test_cats_dir = os.path.join(test_dir, "cat")
test_dogs_dir = os.path.join(test_dir, "dog")

# 打印数据集样本数量校验
print("===== 数据集目录与样本数量校验 =====")
print(f"训练集猫图片：{len(os.listdir(train_cats_dir))} 张")
print(f"训练集狗图片：{len(os.listdir(train_dogs_dir))} 张")
print(f"验证集猫图片：{len(os.listdir(val_cats_dir))} 张")
print(f"验证集狗图片：{len(os.listdir(val_dogs_dir))} 张")
print(f"测试集猫图片：{len(os.listdir(test_cats_dir))} 张")
print(f"测试集狗图片：{len(os.listdir(test_dogs_dir))} 张")

# ========== 2. 数据增强配置 + 增强效果可视化 ==========
# 训练集增强策略
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
# 验证、测试集仅归一化，不做数据增强
val_test_datagen = ImageDataGenerator(rescale=1. / 255)

# 可视化数据增强效果
demo_img_path = os.path.join(train_cats_dir, os.listdir(train_cats_dir)[0])
demo_img = image.load_img(demo_img_path, target_size=(150, 150))
demo_x = image.img_to_array(demo_img)
demo_x = np.expand_dims(demo_x, axis=0)

plt.figure(figsize=(10, 4))
img_count = 0
for batch in train_datagen.flow(demo_x, batch_size=1):
    plt.subplot(1, 4, img_count + 1)
    plt.imshow(image.array_to_img(batch[0]))
    plt.axis("off")
    img_count += 1
    if img_count >= 4:
        break
plt.suptitle("Data Augmentation Sample Images")
plt.show()

# ========== 3. 构建批量数据生成器 ==========
BATCH_SIZE = 16
IMG_SIZE = (150, 150)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)
validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)
test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False  # 测试集不打乱，方便后续对比预测标签
)

# ========== 4. 自建深度Deep CNN模型（调参后最优结构） ==========
# L2正则化系数
reg = regularizers.l2(1e-4)
model_cnn = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    # 卷积块1
    layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # 卷积块2
    layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # 卷积块3
    layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # 卷积块4
    layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # 全局池化替代Flatten，大幅减少参数
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    # 全连接分类层
    layers.Dense(256, activation="relu", kernel_regularizer=reg),
    layers.BatchNormalization(),
    # 二分类输出层
    layers.Dense(1, activation="sigmoid")
])

# 打印模型结构参数量总表
print("\n===== Deep CNN 模型结构摘要 =====")
model_cnn.summary()

# 模型编译：二元交叉熵损失 + Adam优化器 + 准确率指标
model_cnn.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["acc"]
)

# ========== 5. 训练回调函数（早停、保存最优、学习率衰减） ==========
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True  # 训练结束自动回滚到最优权重
)
save_best_model = callbacks.ModelCheckpoint(
    filepath="best_deep_cnn.keras",
    monitor="val_acc",
    save_best_only=True,
    verbose=1
)
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
callback_list = [early_stop, save_best_model, lr_scheduler]

# ========== 6. 模型训练 ==========
print("\n===== 开始训练Deep CNN模型 =====")
history_cnn = model_cnn.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callback_list
)

# 保存完整训练完成模型
model_cnn.save("deep_cnn_final.keras")

# ========== 7. 绘制训练/验证 准确率、损失曲线 ==========
acc = history_cnn.history["acc"]
val_acc = history_cnn.history["val_acc"]
loss = history_cnn.history["loss"]
val_loss = history_cnn.history["val_loss"]
epoch_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
# 子图1：准确率曲线
plt.subplot(1, 2, 1)
plt.plot(epoch_range, acc, "bo", label="Training Acc")
plt.plot(epoch_range, val_acc, "b-", label="Validation Acc")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# 子图2：损失曲线
plt.subplot(1, 2, 2)
plt.plot(epoch_range, loss, "bo", label="Training Loss")
plt.plot(epoch_range, val_loss, "b-", label="Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ========== 8. 加载最优模型，在独立测试集评估分类效果 ==========
print("\n===== 加载全程最优模型，执行测试集评估 =====")
best_cnn_model = models.load_model("best_deep_cnn.keras")
test_loss, test_accuracy = best_cnn_model.evaluate(test_generator, steps=len(test_generator))
print(f"测试集准确率 Test Acc: {test_accuracy:.4f}")
print(f"测试集损失 Test Loss: {test_loss:.4f}")

# 可选：输出少量测试样本预测结果对比
predict_results = best_cnn_model.predict(test_generator, steps=len(test_generator))
predict_labels = (predict_results > 0.5).astype(int).flatten()
true_labels = test_generator.classes
print("\n前20张测试图片：预测标签 | 真实标签（0=猫，1=狗）")
for idx in range(20):
    print(f"样本{idx+1}：预测{predict_labels[idx]} | 真实{true_labels[idx]}")