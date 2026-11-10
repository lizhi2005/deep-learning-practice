# 1. 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']    # 黑体
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示问题
# ------------------------------------------------------
# 2. 读取 MNIST 数据集
# ------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 查看数据形状（验证数据集格式）
print("训练集图像形状：", x_train.shape)    # (60000,28,28)
print("测试集图像形状：", x_test.shape)     # (10000,28,28)
print("标签范围：", np.unique(y_train))     # 0~9

# ------------------------------------------------------
# 3. 数据预处理
# ------------------------------------------------------
# 3.1 图像展平 + 归一化到 [0,1]
x_train = x_train.reshape((60000, 28*28)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28*28)).astype('float32') / 255.0

# 3.2 标签 one-hot 编码
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ------------------------------------------------------
# 4. 构建深度神经网络模型（≥3层隐藏层）
# ------------------------------------------------------
model = Sequential([
    # 隐藏层1：256个神经元，ReLU激活
    Dense(256, activation='relu', input_shape=(784,)),
    # 隐藏层2：128个神经元
    Dense(128, activation='relu'),
    # 隐藏层3：64个神经元
    Dense(64, activation='relu'),
    # 输出层：10分类，softmax
    Dense(10, activation='softmax')
])

# 查看模型结构
model.summary()

# ------------------------------------------------------
# 5. 模型编译（最优参数）
# ------------------------------------------------------
model.compile(
    optimizer='adam',        # 优化器：Adam最优
    loss='categorical_crossentropy', # 多分类损失
    metrics=['accuracy']    # 评价指标：准确率
)

# ------------------------------------------------------
# 6. 训练模型
# ------------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=15,          # 训练轮数
    batch_size=128,     # 批次大小
    validation_split=0.1# 用10%训练集做验证
)

# ------------------------------------------------------
# 7. 在测试集上评估
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n测试集准确率: {test_acc:.4f}")  # 可达到 98% ~ 98.5%

# ------------------------------------------------------
# 8. 绘制训练曲线：Loss 和 Accuracy
# ------------------------------------------------------
plt.figure(figsize=(12,4))

# 损失曲线
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('Loss 曲线')
plt.legend()

# 准确率曲线
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('Accuracy 曲线')
plt.legend()

plt.show()

# ------------------------------------------------------
# 9. 绘制数据集中的样本图像
# ------------------------------------------------------
(x_img, _), _ = mnist.load_data()
plt.figure(figsize=(8,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_img[i], cmap='gray')
    plt.axis('off')
plt.suptitle('MNIST 数据集样本图像')
plt.show()