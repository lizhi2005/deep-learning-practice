#实验一：深度学习基础与全连接网络（MLP）｜反向传播、梯度下降

#实验目标

# 1. 理解张量、自动微分、反向传播
# 2. 搭建多层全连接网络（MLP）对手写数字分类
# 3. 训练、观察 loss 下降，测试准确率

# 分步任务

# 1. 导入库，加载 MNIST 数据集，构造 DataLoader（训练集 + 测试集）
# 2. 定义 MLP 模型类：继承`nn.Module`，包含多层线性层 + 激活函数
# 3. 定义损失函数（交叉熵）、优化器（SGD/Adam）
# 4. 写训练循环：前向传播→计算 loss→反向传播→参数更新
# 5. 写测试循环：关闭梯度，在测试集评估精度
# 6. 绘制训练 loss 曲线


# 实现过程
# 1. 环境与基础（1课时）
# Anaconda创建虚拟环境，安装PyTorch（CPU/GPU版）
# Jupyter Notebook配置与使用
# 张量创建、索引、切片、运算、设备迁移
# 自动求导：requires_grad、backward()、梯度查看



# 2. 从零实现神经网络（1课时）
# 使用NumPy实现两层全连接网络
# 手动实现Sigmoid激活函数及其导数
# 实现前向传播：Z1 = XW1 + b1, A1 = Sigmoid(Z1), Z2 = A1W2 + b2
# 实现反向传播：计算梯度并更新参数
# 在简单数据集上验证算法正确性
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


epochs=10000
lr=0.8

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

Y = np.array([[0, 1, 1, 0]])
m = X.shape[1]
n_x=X.shape[0]
n_h=4
n_y=Y.shape[0]

np.random.seed(0)
W1=np.random.randn(n_h,n_x)*0.01
W2=np.random.randn(n_y,n_h)*0.01
b1=np.random.randn(n_h,1)
b2=np.random.randn(n_y,1)

for i in range(epochs):
    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    loss = np.mean(np.square(A2 - Y))

    dZ2 = (A2 - Y) * sigmoid_derivative(A2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

print("预测值 A2：\n", A2)
print("真实标签 Y：\n", Y)


# 3. PyTorch实现MLP（1.5课时）
# 加载MNIST数据集，进行标准化
# 定义MLP网络：784→256→128→10
# 使用CrossEntropyLoss和Adam优化器
# 编写训练循环，记录损失和准确率
# 绘制训练曲线，在测试集上评估
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt

# 归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# shuffle=True：训练集打乱样本，防止模型记住顺序

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 网络层
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = x.view(-1, 784)
        # 展平层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP()
# 损失函数
loss = nn.CrossEntropyLoss()
# 优化器
opt = optim.Adam(model.parameters(), lr=0.001)

train_loss_list = []
epoch_num = 50


for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        opt.zero_grad()    # 梯度清零
        outputs = model(images)  # 前向传播
        l = loss(outputs, labels) # 计算损失
        l.backward()          # 反向传播，自动求导
        opt.step()         # 更新参数
        running_loss += l.item()

    # 本轮平均训练损失
    avg_loss = running_loss / len(train_loader)
    train_loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f} %")






# 4. 超参数对比（0.5课时）
# 学习率对比：0.001 vs 0.01

# 激活函数对比：Sigmoid vs ReLU
# 记录并分析对比结果

# 学习率对比
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt

# 归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# shuffle=True：训练集打乱样本，防止模型记住顺序

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 网络层
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = x.view(-1, 784)
        # 展平层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP()
# 损失函数
loss = nn.CrossEntropyLoss()
# 优化器

opt1 = optim.Adam(model.parameters(), lr=0.01)


train_loss_list = []
epoch_num = 50


for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        opt1.zero_grad()    # 梯度清零
        outputs = model(images)  # 前向传播
        l = loss(outputs, labels) # 计算损失
        l.backward()          # 反向传播，自动求导
        opt1.step()         # 更新参数
        running_loss += l.item()

    # 本轮平均训练损失
    avg_loss = running_loss / len(train_loader)
    train_loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f} %")
# 批大小对比：32 vs 128

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt

# 归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# batch_size批大小
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# shuffle=True：训练集打乱样本，防止模型记住顺序

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 网络层
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = x.view(-1, 784)
        # 展平层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP()
# 损失函数
loss = nn.CrossEntropyLoss()
# 优化器
opt = optim.Adam(model.parameters(), lr=0.001)

train_loss_list = []
epoch_num = 50


for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        opt.zero_grad()    # 梯度清零
        outputs = model(images)  # 前向传播
        l = loss(outputs, labels) # 计算损失
        l.backward()          # 反向传播，自动求导
        opt.step()         # 更新参数
        running_loss += l.item()

    # 本轮平均训练损失
    avg_loss = running_loss / len(train_loader)
    train_loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f} %")



# （四）考核要求
# 提交NumPy实现的反向传播代码
# 提交PyTorch MLP代码及训练曲线
# 实验报告包含超参数对比分析

















