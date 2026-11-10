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
# 数据集路径
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

# VGG专属回调函数
early_stop_vgg = EarlyStopping(
    monitor='val_acc',
    patience=12,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_vgg = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)
callbacks_vgg = [early_stop_vgg, reduce_lr_vgg]
#加载VGG16预训练主干并冻结
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.trainable = False
# 搭建顶层分类网络
model_vgg = models.Sequential([
    conv_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
# 冻结阶段编译与训练
model_vgg.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
print("\n========== VGG16 冻结主干训练阶段 ==========")
model_vgg.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=callbacks_vgg
)
# 解冻后8层进行微调
conv_base.trainable = True
# 冻结前大半层，仅解冻最后8层
for layer in conv_base.layers[:-8]:
    layer.trainable = False
# 微调阶段使用更小学习率
model_vgg.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)
print("\n VGG16 微调训练阶段")
history_vgg = model_vgg.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=callbacks_vgg
)
#测试集评估
print("\n" + "="*60)
test_loss, test_acc = model_vgg.evaluate(test_gen)
print(f"【VGG16迁移学习】测试集准确率: {test_acc:.4f}")
print(f"【VGG16迁移学习】测试集损失: {test_loss:.4f}")
print("="*60)
#  绘制训练曲线
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
    plt.plot(epochs, val_loss, label='损失曲线')
    plt.title(title + ' 损失曲线')
    plt.tight_layout()
    plt.show()
plot_train_curve(history_vgg, "VGG16迁移学习模型")


















# import os
# import matplotlib.pyplot as plt
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
#
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras import layers, models
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications import VGG16
#
# # ===================== 数据集路径（和你本地路径完全匹配） =====================
# base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
# train_dir = os.path.join(base_dir, "train")
# val_dir = os.path.join(base_dir, "validation")
# test_dir = os.path.join(base_dir, "test")
#
# # ===================== 图像数据增强与归一化 =====================
# # 训练集：开启全部图像增强操作
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# # 验证/测试集：仅像素归一化，不做数据增强
# val_test_datagen = ImageDataGenerator(rescale=1./255)
#
# # 生成批次数据流，二分类binary标签
# train_gen = train_datagen.flow_from_directory(
#     train_dir, target_size=(150,150), batch_size=32, class_mode='binary'
# )
# val_gen = val_test_datagen.flow_from_directory(
#     val_dir, target_size=(150,150), batch_size=32, class_mode='binary'
# )
# test_gen = val_test_datagen.flow_from_directory(
#     test_dir, target_size=(150,150), batch_size=32, class_mode='binary', shuffle=False
# )
#
# # ===================== VGG专用训练回调函数 =====================
# early_stop_vgg = EarlyStopping(
#     monitor='val_acc',
#     patience=12,
#     restore_best_weights=True,
#     verbose=1
# )
# reduce_lr_vgg = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=4,
#     min_lr=1e-7,
#     verbose=1
# )
# vgg_callbacks = [early_stop_vgg, reduce_lr_vgg]
#
# # ===================== 步骤1：加载VGG16预训练主干，冻结全部权重 =====================
# # 载入ImageNet预训练权重，去掉顶层分类头，输入150*150三通道图片
# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
# # 冻结VGG卷积主干，第一轮训练不更新预训练参数
# conv_base.trainable = False
#
# # ===================== 步骤2：搭建完整VGG迁移学习模型 =====================
# model_vgg = models.Sequential([
#     conv_base,                          # 预训练VGG特征提取主干
#     layers.GlobalAveragePooling2D(),     # 全局平均池化替代Flatten，减少参数量
#     layers.Dense(512, activation='relu'),
#     layers.Dropout(0.5),                # Dropout抑制过拟合
#     layers.Dense(1, activation='sigmoid')# 二分类输出，0=猫，1=狗
# ])
#
# # 模型编译：冻结阶段使用较大学习率
# model_vgg.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
#
# # ===================== 第一阶段训练：冻结VGG主干，仅训练自定义分类头 =====================
# print("\n========== VGG16 冻结卷积层 训练阶段 ==========")
# model_vgg.fit(
#     train_gen,
#     epochs=20,
#     validation_data=val_gen,
#     callbacks=vgg_callbacks
# )
#
# # ===================== 步骤3：解冻VGG后8层卷积，微调全部网络 =====================
# conv_base.trainable = True
# # 冻结前半部分卷积层，仅解冻最后2个卷积块（共8层）参与微调
# for layer in conv_base.layers[:-8]:
#     layer.trainable = False
#
# # 微调阶段编译：极小学习率，避免破坏预训练通用特征
# model_vgg.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss='binary_crossentropy',
#     metrics=['acc']
# )
#
# # ===================== 第二阶段训练：微调解冻卷积层+分类头 =====================
# print("\n========== VGG16 解冻后端卷积层 微调阶段 ==========")
# history_vgg = model_vgg.fit(
#     train_gen,
#     epochs=15,
#     validation_data=val_gen,
#     callbacks=vgg_callbacks
# )
#
# # ===================== 测试集模型性能评估 =====================
# print("\n" + "="*60)
# loss_vgg, acc_vgg = model_vgg.evaluate(test_gen)
# print(f"【VGG16迁移学习模型】测试集准确率: {acc_vgg:.4f}")
# print(f"【VGG16迁移学习模型】测试集损失值: {loss_vgg:.4f}")
# print("="*60)
#
# # ===================== 绘制训练准确率/损失曲线 =====================
# def plot_train_curve(history, title):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(len(acc))
#     plt.figure(figsize=(12,4))
#     # 准确率子图
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, acc, label='训练准确率')
#     plt.plot(epochs, val_acc, label='验证准确率')
#     plt.title(title + ' 准确率变化曲线')
#     plt.legend()
#     # 损失子图
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, loss, label='训练损失')
#     plt.plot(epochs, val_loss, label='验证损失')
#     plt.title(title + ' 损失变化曲线')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
# # 绘制VGG完整训练曲线
# plot_train_curve(history_vgg, "VGG16迁移学习模型")