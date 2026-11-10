# 实验目的
# 1.理解卷积运算的数学原理
# 2.掌握卷积层、池化层的工作机制
# 3.能够手动实现2D卷积操作
# 4.能够构建CNN完成图像分类

# 二维卷积：卷积核（kernel）在输入图像矩阵上按照指定步长
# stride 滑动，对应位置元素相乘后求和，得到输出特征图的一个像素值。

# 输出=（输入size+2*填充-卷积size）/步长  +1


# 卷积核提取局部特征：边缘、线条、纹理；
# 权值共享减少参数量；局部感受野，只关注图像一小块区域。

### 池化层（下采样）

# 最常用：最大池化 MaxPool / 平均池化 AvgPool
#
# 池化窗口在特征图滑动，取窗口最大值 / 平均值作为输出
#  不包含可学习参数
# 1. 降低特征图分辨率，减少计算量、参数量
# 2. 保留主要特征，增加特征鲁棒性，轻微平移不变
# 3. 防止过拟合


# 能够手动实现2D卷积操作

import numpy as np

def conv2d_manual(input_img, kernel, stride=1, padding=0):
    h_in,w_in = input_img.shape
    k_h,k_w = kernel.shape
    input_pad=np.pad(input_img,pad_width=padding,mode='constant',constant_values=0)
    h_pad,w_pad = input_pad.shape
    h_out = (h_pad - k_h) // stride + 1
    w_out = (h_pad - k_h) // stride + 1
    output = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            window=input_pad[i*stride:i*stride+k_h,j*stride:j*stride+k_w]
            output[i,j] = np.sum(kernel * window)

    return output

if __name__ == "__main__":
    img = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    res = conv2d_manual(img, kernel, stride=1, padding=1)
    print(res)
# 4.能够构建CNN完成图像分类
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
# 卷积层1  卷积层2  激活函数   池化层   全连接层1  全连接层
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# conv1→relu→pool → conv2→relu→pool → view展平 → fc1+relu → fc2（无激活）→ return logits

model = CNN()
loss_fn = nn.CrossEntropyLoss()
opt= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loss_list = []
test_acc_list = []
epoch_num = 10

for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        opt.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_loss_list.append(avg_loss)

    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            outputs = model(images)
            _,pred = torch.max(outputs,1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    acc = 100 * correct / total
    test_acc_list.append(acc)
    print(f"Epoch {epoch+1}, Train Loss:{avg_loss:.4f}, Test Acc:{acc:.2f}%")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="Train Loss")
plt.xlabel("Epoch")
plt.title("CNN Training Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(test_acc_list, color="orange", label="Test Acc")
plt.xlabel("Epoch")
plt.title("CNN Test Accuracy")
plt.legend()
plt.show()









# （二）实验内容
# 序号	内容	课时
# 1	手动实现2D卷积和池化（NumPy）	1
# 2	构建简单CNN完成CIFAR-10分类	1.5
# 3	卷积核可视化与特征图分析	1
# 4	不同CNN架构对比实验	0.5



# （三）实现过程
# 1. 手动实现卷积（1课时）
# 实现2D卷积：滑动窗口、逐元素相乘求和
# 实现不同步长（stride=1,2）和填充（padding=0,1）
# 实现最大池化和平均池化
# 与PyTorchConv2d结果对比验证
import numpy as np

def conv2d_manual(input_img, kernel, stride=1, padding=0):
    h_in,w_in = input_img.shape
    k_h,k_w = kernel.shape
    input_pad=np.pad(input_img,pad_width=padding,mode='constant',constant_values=0)
    h_pad,w_pad = input_pad.shape
    h_out = (h_pad - k_h) // stride + 1
    w_out = (h_pad - k_h) // stride + 1
    output = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            window=input_pad[i*stride:i*stride+k_h,j*stride:j*stride+k_w]
            output[i,j] = np.sum(kernel * window)

    return output

def maxpool2d_manual(input_mat, kernel_size=2, stride=2):
    h_in, w_in = input_mat.shape
    k = kernel_size
    h_out = (h_in - k) // stride + 1
    w_out = (w_in - k) // stride + 1
    out = np.zeros((h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            window = input_mat[i*stride:i*stride+k, j*stride:j*stride+k]
            out[i,j] = np.max(window)
    return out

def avgpool2d_manual(input_mat, kernel_size=2, stride=2):
    h_in, w_in = input_mat.shape
    k = kernel_size
    h_out = (h_in - k) // stride + 1
    w_out = (w_in - k) // stride + 1
    out = np.zeros((h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            window = input_mat[i*stride:i*stride+k, j*stride:j*stride+k]
            out[i,j] = np.mean(window)
    return out

# 最大池化 MaxPool：在滑动窗口里，取窗口中像素的最大值作为输出。
# 平均池化 AvgPool：在滑动窗口里，计算窗口所有像素的平均值作为输出。

# 最大池化
# 1. 只保留区域内最强、最显著的特征，弱化弱信号。
# 2. 对纹理、边缘特征提取效果更好。
# 3. 抗轻微形变、平移能力更强；但容易丢失部分弱特征信息。
#
# 平均池化
# 1. 对窗口内特征做平滑，把整体信息保留下来。
# 2. 减少噪声带来的剧烈波动，更关注全局信息。
# 3. 会把特征 “稀释”，边缘、纹理对比度会降低

if __name__ == "__main__":
    input_np = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.float32)

    kernel_np = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    conv_manual1 = conv2d_manual(input_np, kernel_np, stride=1, padding=0)
    print('stride=1, padding=0:',conv_manual1)

    conv_manual2 = conv2d_manual(input_np, kernel_np, stride=2, padding=1)
    print('stride=2, padding=1:',conv_manual2)

    max_man = maxpool2d_manual(input_np, 2, 2)
    print('最大池化:',max_man)



    avg_man = avgpool2d_manual(input_np, 2, 2)
    print('平均池化:',avg_man)
# 与PyTorchConv2d结果对比验证

#  测试stride=1, padding=0
# numpy转pytorch张量，扩展为B,C,H,W四维格式
x_torch = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0)
# 构建pytorch卷积层，in_channel=1 out_channel=1, 3*3卷积核，关闭bias
conv_pt1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
# 将我们自己定义的卷积核赋值给pytorch卷积权重
conv_pt1.weight.data = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0)
# 前向计算，去掉多余维度转回numpy
conv_pt_out1 = conv_pt1(x_torch).squeeze().detach().numpy()

print("PyTorch Conv2d结果：")
print(conv_pt_out1)

# 测试stride=2, padding=1
conv_pt2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
conv_pt2.weight.data = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0)
conv_pt_out2 = conv_pt2(x_torch).squeeze().detach().numpy()

print("PyTorch Conv2d结果：")
print(conv_pt_out2)




# 2. 构建CNN（1.5课时）
# 网络结构：
# oConv1: 3→32, 3×3, padding=1 → ReLU → MaxPool(2)
# oConv2: 32→64, 3×3, padding=1 → ReLU → MaxPool(2)
# oFC1: 64×8×8→256 → ReLU
# oFC2: 256→10
# 加载CIFAR-10数据集，进行归一化
# 训练模型，记录损失和准确率
# 测试集评估，绘制混淆矩阵

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


cifar_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
cifar_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8) #展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CifarCNN()
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epoch_num = 10
train_loss_list = []
test_acc_list = []

for epoch in range(epoch_num):
    model.train()
    # 用训练集数据，更新网络权重，降低损失
    running_loss = 0.0
    for images, labels in cifar_train_loader:
        opt.zero_grad()
        # 梯度清零
        outputs = model(images)
        # 前向传播
        loss = loss_fn(outputs, labels)
        # 计算损失
        loss.backward()
        # 反向传播
        opt.step()
        # 参数更新
        running_loss += loss.item()

    avg_loss = running_loss / len(cifar_train_loader)
    train_loss_list.append(avg_loss)

    model.eval()
    # 在测试集上评估模型准确率，不更新参数，顺便收集预测标签，用来后面画混淆矩阵
    correct = 0
    total = 0
    all_preds = []
    # 保存所有模型预测结果
    all_labels = []
    # 保存所有真实标签
    with torch.no_grad():
        for images, labels in cifar_test_loader:
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = 100 * correct / total
    test_acc_list.append(acc)
    print(f"Epoch {epoch+1}, Train Loss:{avg_loss:.4f}, Test Acc:{acc:.2f}%")


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="Train Loss")
plt.xlabel("Epoch")
plt.title("CIFAR10 Train Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(test_acc_list, color="orange", label="Test Acc")
plt.xlabel("Epoch")
plt.title("CIFAR10 Test Accuracy")
plt.legend()
plt.show()

# 5. 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("CIFAR10 混淆矩阵")
plt.show()


# 3. 可视化分析（1课时）
# 可视化第一层卷积核（权重）
# 可视化中间层特征图
# 分析不同层提取的特征（边缘→纹理→语义）
# 使用Grad-CAM可视化模型关注区域（选做）



# 4. 架构对比（0.5课时）
# 对比不同卷积核大小（3×3 vs 5×5）
# 对比不同网络深度（2层 vs 3层卷积）
# 记录并分析对比结果
# （四）考核要求
# 提交手动实现的卷积和池化代码
# 提交CNN模型代码及训练结果
# 实验报告包含特征图可视化分析
