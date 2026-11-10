# 放在最顶部，关闭 oneDNN 警告
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# ===================== 1. 统一配置正确路径 =====================
BASE_PATH = r"D:\pycharm-project\PythonProject1\shenduxuexi-shiyan-1\cats_and_dogs_small"
train_dir = os.path.join(BASE_PATH, "train")
validation_dir = os.path.join(BASE_PATH, "validation")
test_dir = os.path.join(BASE_PATH, "test")

# 训练集猫文件夹路径
train_cats_dir = os.path.join(train_dir, "cat")

# 路径&数据数量检查
print("=== 数据集路径检查 ===")
print("数据集根目录:", BASE_PATH)
print("训练集目录:", train_dir)
print("训练集猫图片数量:", len(os.listdir(train_cats_dir)))
print("训练集狗图片数量:", len(os.listdir(os.path.join(train_dir, "dog"))))
print("验证集猫图片数量:", len(os.listdir(os.path.join(validation_dir, "cat"))))
print("验证集狗图片数量:", len(os.listdir(os.path.join(validation_dir, "dog"))))
print("测试集猫图片数量:", len(os.listdir(os.path.join(test_dir, "cat"))))
print("测试集狗图片数量:", len(os.listdir(os.path.join(test_dir, "dog"))))

# ===================== 2. 数据增强可视化 =====================
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 读取一张猫的图片做增强演示
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[0]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

plt.figure(figsize=(10, 4))
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(1, 4, i+1)
    plt.imshow(image.array_to_img(batch[0]))
    plt.axis('off')
    i += 1
    if i % 4 == 0:
        break
plt.suptitle("Data Augmentation Examples")
plt.show()

# ===================== 3. 搭建CNN模型 =====================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc']
)

# ===================== 4. 构建数据生成器 =====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ===================== 5. 模型训练 =====================
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

model.save('cats_and_dogs_small_2.h5')

# ===================== 6. 绘制训练曲线 =====================
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# ===================== 7. 测试集评估 =====================
print("\n=== 测试集评估 ===")
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")




























# # 统一导入：全部使用 tensorflow.keras (TF2 标准)
# import os
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, models, optimizers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
#
# # ===================== 1. 配置数据集路径 (务必修改为你本地真实路径) =====================
# train_dir = r'D:\kaggle\modle_date\train'
# validation_dir = r'D:\kaggle\modle_date\validation'
# train_cats_dir = r'D:\modle_date\train\cats'
#
# # ===================== 2. 数据增强可视化 (查看增强效果) =====================
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# # 读取单张图片做增强演示
# fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
# img_path = fnames[3]
# img = image.load_img(img_path, target_size=(150, 150))
# x = image.img_to_array(img)
# x = x.reshape((1,) + x.shape)
#
# # 生成4张增强图片
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break
# plt.show()
#
# # ===================== 3. 搭建CNN模型 (加入Dropout防过拟合) =====================
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dropout(0.5),  # 随机丢弃50%神经元，抑制过拟合
#     layers.Dense(512, activation='relu'),
#     layers.Dense(1, activation='sigmoid')  # 二分类输出
# ])
#
# # 打印网络结构
# model.summary()
#
# # 模型编译 (修正lr为learning_rate，英文引号)
# model.compile(
#     loss='binary_crossentropy',
#     optimizer=optimizers.RMSprop(learning_rate=1e-4),
#     metrics=['acc']
# )
#
# # ===================== 4. 构建数据生成器 & 数据流 =====================
# # 训练集：数据增强 + 归一化
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
#
# # 验证集：仅归一化，不做数据增强
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary'
# )
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='binary'
# )
#
# # ===================== 5. 模型训练 (废弃fit_generator，改用标准fit) =====================
# history = model.fit(
#     train_generator,
#     steps_per_epoch=63,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=32
# )
#
# # 保存模型
# model.save('cats_and_dogs_small_2.h5')
#
# # ===================== 6. 绘制训练曲线 (准确率 & 损失) =====================
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
#
# # 准确率曲线
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
#
# # 损失曲线
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()