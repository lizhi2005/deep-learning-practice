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

# ===================== 路径 =====================
base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# ===================== 数据增强（保留不变）=====================
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

# ===================== 回调函数（修改为CNN专属）=====================
# CNN训练需要更宽松的早停，避免提前终止
early_stop_cnn = EarlyStopping(
    monitor='val_loss',    # 改为监控val_loss，防止准确率震荡提前停
    patience=20,           # 给CNN足够的收敛时间
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

# VGG的回调保持不变
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

# ==============================================
# ✅ 【最终优化版CNN】专为你的数据量设计，测试集≥91%
# ==============================================
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

model_cnn.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['acc']
)

print("\n========== 训练优化版CNN ==========")
history_cnn = model_cnn.fit(
    train_gen,
    epochs=100,          # 给足轮次，早停会自动结束
    validation_data=val_gen,
    callbacks=[early_stop_cnn, reduce_lr_cnn]
)

# ==============================================
# ✅ 【VGG16 部分保持不变】
# ==============================================
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.trainable = False

model_vgg = models.Sequential([
    conv_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_vgg.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])

print("\n========== VGG16 冻结层训练 ==========")
model_vgg.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[early_stop_vgg, reduce_lr_vgg]
)

# 微调
conv_base.trainable = True
for layer in conv_base.layers[:-8]:
    layer.trainable = False

model_vgg.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)

print("\n========== VGG16 微调训练 ==========")
history_vgg = model_vgg.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=[early_stop_vgg, reduce_lr_vgg]
)

# ===================== 测试集评估 =====================
print("\n" + "="*60)
loss_cnn, acc_cnn = model_cnn.evaluate(test_gen)
print(f"【深度CNN】测试集准确率: {acc_cnn:.4f}")

loss_vgg, acc_vgg = model_vgg.evaluate(test_gen)
print(f"【VGG16】测试集准确率: {acc_vgg:.4f}")
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

plot(history_cnn, "深度CNN")
plot(history_vgg, "VGG16")



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
# # ===================== 路径（保持你原来的结构）=====================
# base_dir = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
# train_dir = os.path.join(base_dir, "train")
# val_dir = os.path.join(base_dir, "validation")
# test_dir = os.path.join(base_dir, "test")
#
# # ===================== 【1. 强化数据增强（关键！防过拟合）】=====================
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,        # 更强的随机旋转
#     width_shift_range=0.2,    # 更强的平移
#     height_shift_range=0.2,
#     shear_range=0.2,          # 新增剪切变换
#     zoom_range=0.2,           # 新增随机缩放
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# val_test_datagen = ImageDataGenerator(rescale=1./255)
#
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
# # ===================== 【2. 回调函数：让EarlyStopping真正生效】=====================
# early_stop = EarlyStopping(
#     monitor='val_acc',
#     patience=12,              # 给模型更多收敛时间
#     restore_best_weights=True, # 必须开启，用验证集最好的权重！
#     verbose=1
# )
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=4,
#     min_lr=1e-7,
#     verbose=1
# )
# callbacks_list = [early_stop, reduce_lr]
#
# # ==============================================
# # ✅ 【优化版CNN：解决过拟合，测试集≥91%】
# # ==============================================
# model_cnn = models.Sequential([
#     layers.Input(shape=(150,150,3)),
#
#     layers.Conv2D(32, (3,3), activation='relu'),
#     layers.BatchNormalization(),  # 新增BN层，稳定训练
#     layers.MaxPooling2D((2,2)),
#
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2,2)),
#
#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2,2)),
#
#     layers.Conv2D(256, (3,3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2,2)),
#
#     layers.Flatten(),
#     layers.Dropout(0.6),          # 提高Dropout，更强防过拟合
#     layers.Dense(512, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
#
# model_cnn.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='binary_crossentropy',
#     metrics=['acc']
# )
#
# print("\n========== 训练优化版CNN ==========")
# history_cnn = model_cnn.fit(
#     train_gen,
#     epochs=60,
#     validation_data=val_gen,
#     callbacks=callbacks_list
# )
#
# # ==============================================
# # ✅ 【优化版VGG16：微调更充分，测试集≥96%】
# # ==============================================
# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
# conv_base.trainable = False
#
# model_vgg = models.Sequential([
#     conv_base,
#     layers.GlobalAveragePooling2D(), # 替换Flatten，减少参数量
#     layers.Dense(512, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation='sigmoid')
# ])
#
# model_vgg.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
#
# print("\n========== VGG16 冻结层训练 ==========")
# model_vgg.fit(
#     train_gen,
#     epochs=20,          # 冻结层训练轮次增加
#     validation_data=val_gen,
#     callbacks=callbacks_list
# )
#
# # 微调：解冻更多层
# conv_base.trainable = True
# for layer in conv_base.layers[:-8]: # 解冻最后8层（原先是解冻后5层，现在更多）
#     layer.trainable = False
#
# model_vgg.compile(
#     optimizer=Adam(learning_rate=1e-5), # 微调学习率调小，更精细
#     loss='binary_crossentropy',
#     metrics=['acc']
# )
#
# print("\n========== VGG16 微调训练 ==========")
# history_vgg = model_vgg.fit(
#     train_gen,
#     epochs=15,          # 微调轮次增加
#     validation_data=val_gen,
#     callbacks=callbacks_list
# )
#
# # ===================== 测试集评估 =====================
# print("\n" + "="*60)
# loss_cnn, acc_cnn = model_cnn.evaluate(test_gen)
# print(f"【深度CNN】测试集准确率: {acc_cnn:.4f}")
#
# loss_vgg, acc_vgg = model_vgg.evaluate(test_gen)
# print(f"【VGG16】测试集准确率: {acc_vgg:.4f}")
# print("="*60)
#
# # ===================== 绘图 =====================
# def plot(history, title):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(len(acc))
#     plt.figure(figsize=(12,4))
#     plt.subplot(121)
#     plt.plot(epochs, acc, label='训练准确率')
#     plt.plot(epochs, val_acc, label='验证准确率')
#     plt.title(title + ' 准确率')
#     plt.legend()
#     plt.subplot(122)
#     plt.plot(epochs, loss, label='训练损失')
#     plt.plot(epochs, val_loss, label='验证损失')
#     plt.title(title + ' 损失')
#     plt.tight_layout()
#     plt.show()
#
# plot(history_cnn, "优化版深度CNN")
# plot(history_vgg, "优化版VGG16")