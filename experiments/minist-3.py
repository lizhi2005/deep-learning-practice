import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2  # 这里修复！

# 1. 读取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 数据预处理
x_train = x_train.reshape((60000, 28 * 28)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28 * 28)).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. 改进模型（加入Dropout和正则化防过拟合）
model = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.summary()

# 4. 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 早停
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 6. 训练
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 7. 测试集评估
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n测试集准确率: {test_acc:.4f}")

# 8. 绘制曲线
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