# 1. 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭烦人的oneDNN警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 关闭TF多余日志

# 解决 matplotlib 中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# ------------------------------------------------------
# 2. 读取 MNIST 数据集
# ------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("训练集图像形状：", x_train.shape)
print("测试集图像形状：", x_test.shape)
print("标签范围：", np.unique(y_train))

# ------------------------------------------------------
# 3. 数据预处理
# ------------------------------------------------------
x_train = x_train.reshape((60000, 28 * 28)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28 * 28)).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ------------------------------------------------------
# 4. 构建深度神经网络模型（无警告版）
# ------------------------------------------------------
model = Sequential([
    Input(shape=(784,)),  # 用Input层，消除input_shape警告
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

# ------------------------------------------------------
# 5. 编译模型
# ------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------------------
# 6. 训练模型
# ------------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ------------------------------------------------------
# 7. 测试集评估
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n测试集准确率: {test_acc:.4f}")

# ------------------------------------------------------
# 8. 绘制 Loss & Acc 曲线（中文正常）
# ------------------------------------------------------
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

# ------------------------------------------------------
# 9. 绘制数据集样本图
# ------------------------------------------------------
(x_img, _), _ = mnist.load_data()
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_img[i], cmap='gray')
    plt.axis('off')
plt.suptitle('MNIST 手写数字样本')
plt.show()