# 库导入与环境配置
import numpy as np
import matplotlib.pyplot as plt
import os
# 导入深度学习所需模块
from tensorflow.keras.datasets import mnist                    # MNIST手写数字数据集
from tensorflow.keras.models import Sequential                 # 顺序神经网络模型
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten  # 网络层
from tensorflow.keras.utils import to_categorical             # 标签独热编码
from tensorflow.keras.callbacks import EarlyStopping           # 早停法，防止过拟合
from tensorflow.keras.regularizers import l2                   # L2正则化，抑制过拟合
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 图像增强
# 关闭TensorFlow不必要的警告信息，保持控制台整洁
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置Matplotlib中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False




# ===================== 1. 加载MNIST数据集 =====================
# 加载训练集与测试集，x为图像数据，y为对应标签(0~9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ===================== 2. 图像可视化：原始图→切片→展平 =====================
# 创建画布，用于展示图像预处理过程
plt.figure(figsize=(14, 5))

# 子图1：展示第1000张原始图像
plt.subplot(1, 3, 1)
sample_img = x_train[1000]
plt.imshow(sample_img, cmap='gray')
plt.title(f"原始样本 - 数字{y_train[1000]}")
plt.axis('off')

# 子图2：图像中心切片，提取数字核心区域
plt.subplot(1, 3, 2)
patch = sample_img[8:20, 8:20]  # 从28×28中截取中心12×12区域
plt.imshow(patch, cmap='gray')
plt.title("图像中心切片 (8:20,8:20)")
plt.axis('off')

# 子图3：将二维图像展平为一维向量（全连接网络输入要求）
plt.subplot(1, 3, 3)
plt.imshow(sample_img.reshape(784, 1), cmap='gray', aspect='auto')
plt.title("展平为 784 维向量")
plt.axis('off')

plt.suptitle("原始图像 → 切片 → 展平全过程", fontsize=14)
plt.tight_layout()
plt.show()


# ===================== 3. 数据预处理 =====================
# 将图像数据增加通道维度，并归一化到 [0,1] 区间，提升训练稳定性
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# 标签进行独热编码（One-Hot），适配多分类损失函数
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 从训练集划分6000个样本作为验证集，用于监控训练过程
val_size = 6000
x_val = x_train[-val_size:]
y_val = y_train[-val_size:]
x_train = x_train[:-val_size]
y_train = y_train[:-val_size]


# ===================== 4. 图像增强配置 =====================
# 定义图像增强器：随机旋转、平移、缩放，扩充数据多样性，减少过拟合
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(x_train)  # 基于训练集数据统计信息适配增强器


# ===================== 5. 图像增强效果可视化 =====================
def show_augmented_img(img, datagen, n=6):
    """
    展示单张图像增强前后对比效果
    :param img: 输入图像
    :param datagen: 图像增强器
    :param n: 生成增强图像数量
    """
    img = img.reshape(1, 28, 28, 1)
    plt.figure(figsize=(12, 3))

    # 显示原始图像
    plt.subplot(1, n + 1, 1)
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title("原始图像")
    plt.axis('off')

    # 循环生成并显示增强图像
    i = 2
    for batch in datagen.flow(img, batch_size=1):
        plt.subplot(1, n + 1, i)
        plt.imshow(batch[0, :, :, 0], cmap='gray')
        plt.title(f"增强{i - 1}")
        plt.axis('off')
        i += 1
        if i > n + 1:
            break
    plt.suptitle("图像增强效果对比", fontsize=14)
    plt.show()

# 展示第1000张图像的增强效果
show_augmented_img(x_train[1000], datagen)


# ===================== 6. 模型构建函数 =====================
def create_model():
    """
    构建全连接神经网络模型
    包含：展平层 + 3层隐藏层 + Dropout + 输出层
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),               # 输入层：28×28灰度图
        Flatten(),                             # 展平为784维向量
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),  # 第一层隐藏层
        Dropout(0.3),                          # 随机丢弃30%神经元，防止过拟合
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),  # 第二层隐藏层
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),  # 第三层隐藏层
        Dropout(0.2),
        Dense(10, activation='softmax')        # 输出层：10分类，softmax激活
    ])
    return model


# ===================== 7. 实验配置：不同优化器 & 损失函数对比 =====================
# 定义四组实验组合，用于对比性能
configs = [
    {"opt": "adam", "loss": "categorical_crossentropy", "name": "Adam + 交叉熵"},
    {"opt": "sgd", "loss": "categorical_crossentropy", "name": "SGD + 交叉熵"},
    {"opt": "rmsprop", "loss": "categorical_crossentropy", "name": "RMSprop + 交叉熵"},
    {"opt": "adam", "loss": "mse", "name": "Adam + MSE"},
]

# 存储每组实验结果
results = []
histories = []


# ===================== 8. 循环训练与评估 =====================
for cfg in configs:
    print("\n========================================")
    print(f"正在训练：{cfg['name']}")
    print("========================================")

    # 构建模型
    model = create_model()

    # 编译模型：指定优化器、损失函数、评估指标
    model.compile(
        optimizer=cfg["opt"],
        loss=cfg["loss"],
        metrics=["accuracy"]
    )

    # 早停法：验证集损失5轮不下降则停止，恢复最优权重
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # 训练模型（使用图像增强）
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # 在测试集上评估最终精度
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results.append({"name": cfg["name"], "acc": test_acc})
    histories.append(history)


# ===================== 9. 输出实验对比表格 =====================
print("\n" + "=" * 50)
print("优化器 & 损失函数 最终测试准确率对比")
print("=" * 50)
for r in results:
    print(f"{r['name']:20s} | 准确率 = {r['acc']:.4f}")


# ===================== 10. 绘制训练曲线对比图 =====================
plt.figure(figsize=(14, 5))

# 子图1：验证集准确率对比
plt.subplot(1, 2, 1)
for i, h in enumerate(histories):
    plt.plot(h.history['val_accuracy'], label=configs[i]['name'])
plt.title("验证集准确率对比")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# 子图2：验证集损失对比
plt.subplot(1, 2, 2)
for i, h in enumerate(histories):
    plt.plot(h.history['val_loss'], label=configs[i]['name'])
plt.title("验证集损失对比")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()