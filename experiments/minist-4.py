import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. 读取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 数据预处理
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 手动切分验证集（修复错误）
val_size = 6000
x_val = x_train[-val_size:]
y_val = y_train[-val_size:]
x_train = x_train[:-val_size]
y_train = y_train[:-val_size]

# 3. 图像增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# 4. 模型：512 → 256 → 128 全连接结构（正确版）
model = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),  # 必须加，把图像展平
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.summary()

# 5. 编译
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. 早停
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 7. 训练（已修复所有错误）
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=30,
    validation_data=(x_val, y_val),  # 这里修复！
    callbacks=[early_stop],
    verbose=1
)

# 8. 测试集评估
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n测试集准确率: {test_acc:.4f}")

# 9. 绘制曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失曲线')
plt.xlabel('训练轮次')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率曲线')
plt.xlabel('训练轮次')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()