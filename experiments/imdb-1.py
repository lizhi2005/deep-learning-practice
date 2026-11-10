# ===================== 导入所需库 =====================
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭烦人的oneDNN警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 关闭TF多余日志

# 解决 matplotlib 中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 关闭多余警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
plt.rcParams["font.sans-serif"] = ["SimHei"]    # 中文显示
plt.rcParams["axes.unicode_minus"] = False

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# ===================== 1. 超参数全局配置 =====================
max_words = 10000       # 选取词汇表前10000个高频词
max_len = 200           # 文本序列统一长度
val_split = 0.2         # 训练集划分20%作为验证集

# ===================== 2. 加载IMDB数据集 =====================
# 加载数据集：文本为整数序列，标签0=负面评价，1=正面评价
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)

print(f"训练集样本数：{len(train_data)}")
print(f"测试集样本数：{len(test_data)}")
print(f"单条文本示例(整数序列)：{train_data[0][:10]}")
print(f"对应标签(0=负面,1=正面)：{train_labels[0]}")

# ===================== 3. 自定义函数：序列转 BOW 词袋向量(One-Hot) =====================
def vectorize_sequences(sequences, dimension=max_words):
    """
    将整数文本序列转为 词袋(BOW) One-Hot 向量
    :param sequences: 文本整数序列列表
    :param dimension: 词汇表维度
    :return: 二维张量 (样本数, 词汇表大小)
    """
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.0  # 对应单词位置置1，实现One-Hot编码
    return results

# 3.1 文本向量化（BOW）
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 3.2 标签向量化（二分类标签直接转为浮点型）
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# 3.3 划分训练集 & 验证集
x_val = x_train[:int(len(x_train)*val_split)]
partial_x_train = x_train[int(len(x_train)*val_split):]
y_val = y_train[:int(len(y_train)*val_split)]
partial_y_train = y_train[int(len(y_train)*val_split):]

# ===================== 4. 构建全连接神经网络函数 =====================
def build_model(hidden_layers=2, units=64, activation="relu"):
    """
    构建全连接情感分类网络
    :param hidden_layers: 隐藏层层数
    :param units: 每层隐藏层神经元数
    :param activation: 隐藏层激活函数
    :return: 神经网络模型
    """
    model = Sequential()
    # 第一层输入层 + 第一个隐藏层
    model.add(Dense(units, activation=activation, input_shape=(max_words,)))
    # 叠加额外隐藏层
    for _ in range(hidden_layers - 1):
        model.add(Dense(units, activation=activation))
    # 输出层：1个神经元，sigmoid激活，输出情感概率(0~1)
    model.add(Dense(1, activation="sigmoid"))
    return model

# ===================== 5. 多组实验配置：调优网络结构、优化器、损失函数 =====================
# 实验组：网络结构 + 优化器 + 损失函数组合
exp_configs = [
    # 组合1：2隐藏层+64神经元+relu + Adam + 二元交叉熵(标准二分类)
    {"layer":2, "unit":64, "act":"relu", "opt":"adam", "loss":"binary_crossentropy", "name":"结构2-64-relu + Adam+交叉熵"},
    # 组合2：3隐藏层+32神经元+relu + SGD + 二元交叉熵
    {"layer":3, "unit":32, "act":"relu", "opt":"sgd", "loss":"binary_crossentropy", "name":"结构3-32-relu + SGD+交叉熵"},
    # 组合3：2隐藏层+64神经元+tanh + Adam + MSE(回归损失对比)
    {"layer":2, "unit":64, "act":"tanh", "opt":"adam", "loss":"mse", "name":"结构2-64-tanh + Adam+MSE"},
]

# 保存实验结果
exp_results = []
history_list = []

# ===================== 6. 循环训练所有实验组 =====================
for cfg in exp_configs:
    print("\n" + "="*60)
    print(f"开始实验：{cfg['name']}")
    print("="*60)

    # 搭建模型
    model = build_model(hidden_layers=cfg["layer"], units=cfg["unit"], activation=cfg["act"])
    # 模型编译：优化器、损失函数、评估指标
    model.compile(optimizer=cfg["opt"], loss=cfg["loss"], metrics=["accuracy"])
    # 早停：验证集loss 5轮不下降则停止，防止过拟合
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # 模型训练
    history = model.fit(
        partial_x_train, partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    history_list.append(history)

    # 测试集评估
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    exp_results.append({
        "name":cfg["name"],
        "test_loss":test_loss,
        "test_acc":test_acc
    })

# ===================== 7. 输出所有实验精度对比 =====================
print("\n" + "="*60)
print("【BOW词袋表示】各组模型测试集精度汇总")
print("="*60)
for res in exp_results:
    print(f"{res['name']:40s} | 测试准确率：{res['test_acc']:.4f}")

# ===================== 8. 绘制 Loss & Accuracy 曲线（最优组曲线，也可绘制全部） =====================
# 选取最优一组（本实验第一组为最优）
best_history = history_list[0]

plt.figure(figsize=(12, 4))
# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(best_history.history["loss"], label="训练损失")
plt.plot(best_history.history["val_loss"], label="验证损失")
plt.title("模型损失变化曲线")
plt.xlabel("训练轮次 Epoch")
plt.ylabel("Loss")
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(best_history.history["accuracy"], label="训练准确率")
plt.plot(best_history.history["val_accuracy"], label="验证准确率")
plt.title("模型准确率变化曲线")
plt.xlabel("训练轮次 Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()