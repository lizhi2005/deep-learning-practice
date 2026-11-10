
import os

data_dir = './data/cat_dog'

# 遍历 train / test 两个划分
total_train = 0
total_test = 0
total_val = 0


for split in ['train', 'test', 'val']:
    split_path = os.path.join(data_dir, split)
    if not os.path.exists(split_path):
        print(f"⚠️ 找不到目录: {split_path}")
        continue

    print(f"\n===== {split} 集 =====")
    # 遍历 split 下的每个类别子文件夹（cat、dog）
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):            # 只统计文件夹（类别）
            n = len(os.listdir(cls_path))      # 该类别图片数
            print(f"  {cls}: {n} 张")
            if split == 'train':
                total_train += n
            elif split == 'test':
                total_test += n
            else:
                total_val+= n

print(f"\n========== 汇总 ==========")
print(f"训练集总样本数: {total_train}")
print(f"测试集总样本数: {total_test}")
print(f"验证集总样本数: {total_val}")

print(f"全部样本数:     {total_train + total_test+total_val}")

#
#
# from modelscope.msdatasets import MsDataset
# from modelscope.utils.constant import DownloadMode
#
# ms_train_dataset = MsDataset.load(
#             'garbage265', namespace='tany0699',
#             subset_name='default', split='train',
#             download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS) # 加载训练集
# print(next(iter(ms_train_dataset)))
#
# ms_val_dataset = MsDataset.load(
#             'garbage265', namespace='tany0699',
#             subset_name='default', split='validation',
#             download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS) # 加载验证集
# print(next(iter(ms_val_dataset)))



# （一）实验目的
# 1.理解BatchNorm和Dropout的原理与作用


# 批归一化是一种对神经网络某一层输入的激活值进行标准化的技术。它对每个训练批次内的数据，
# 沿通道维度计算均值和方差，将激活值归一化为均值为 0、方差为 1 的标准分布
# ，再通过可学习的缩放和平移参数进行重构，从而缓解训练过程中各层输入分布的变化

# 作用：

# 1. 加速收敛：标准化使各层输入分布稳定，可允许使用更大的学习率；
# 2. 缓解梯度问题：避免激活值过大或过小导致梯度消失 / 爆炸；
# 3. 兼具轻微正则化效果：批内统计量引入一定噪声，提升泛化能力；
# 4. 减少对参数初始化和学习率调参的敏感性。

# Dropout 是一种正则化技术，用于缓解神经网络的过拟合。在训练过程中，
# 以一定概率 p 随机 "丢弃"（即暂时置零）部分神经元，使其不参与前向与反向传播；
# 在测试阶段保留全部神经元，并将权重按概率进行缩放，保证输出与训练时一致。


# 作用：
#
# 1. 缓解过拟合：迫使网络不依赖个别神经元，学习到更鲁棒、分布式的特征；
# 2. 类似集成学习效果：每次训练相当于训练不同的子网络，测试时近似对这些子网络的集成；
# 3. 增强泛化能力：尤其在数据量有限、网络规模较大的场景下效果显著。


# 2.掌握数据增强技术

# 在数据加载阶段对训练图像实施随机水平翻转、随机裁剪、随机旋转与色彩抖动，
# 扩充训练样本多样性，配合迁移学习进一步抑制过拟合。


# 3.理解迁移学习的基本原理

# 迁移学习是一种机器学习方法，指将某个源任务（如大规模 ImageNet 分类）上训练好的模型知识，
# 迁移应用到另一个相关但不同的目标任务（如自定义小样本图像分类）上。其核心是借助源任务学到的通用特征，
# 减少目标任务对大量标注数据的依赖，提升在小数据集上的训练效果。

# 4.能够使用预训练模型完成新任务


# （二）实验内容
# 序号	内容	课时
# 1	BatchNorm与Dropout实现与对比	1
# 2	数据增强技术实践	1
# 3	预训练模型加载与特征提取	1
# 4	迁移学习微调与对比实验	1
# （三）实现过程
# 1. BatchNorm与Dropout（1课时）
# 手动实现BatchNorm的前向传播
# 在CNN中添加BatchNorm层，对比训练速度和准确率
# 在CNN中添加Dropout层，对比过拟合情况
# 对照实验：无正则化 / 仅BN / 仅Dropout / BN+Dropout

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# NumPy 手动实现 BatchNorm 前向传播
def batchnorm_forward(x, gamma, beta, eps=1e-5):
    '''
    x：输入数据，形状 (N, C, H, W) —— N 张图、C 个通道、每张 H×W 像素
    gamma：可学习的缩放参数（γ）
    beta：可学习的平移参数（β）
    eps：一个很小的数，防止除以 0

    '''
    N, C, H, W = x.shape
    x_flat = x.transpose(0, 1).reshape(C, -1)
    mu = x_flat.mean(axis=1, keepdims=True)
    var = x_flat.var(axis=1, keepdims=True)
    x_norm = (x_flat - mu) / np.sqrt(var + eps)
    y_flat = gamma.reshape(-1, 1) * x_norm + beta.reshape(-1, 1)
    y = y_flat.reshape(C, N, H, W).transpose(1, 0, 2, 3)
    return y, (x, mu, var, x_norm, gamma, beta)


# 基准 CNN：无 BatchNorm、无 Dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        x = x.view(-1, 64 * 8 * 8)      # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)      # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 加 BatchNorm

class CNN_BN(nn.Module):
    """在每层卷积后加 BatchNorm2d"""
    def __init__(self):
        super(CNN_BN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)          # 对应 conv1 输出通道
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)          # 对应 conv2 输出通道
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)                        # BN 在激活函数之前
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)             # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 加 Dropout
class CNN_Dropout(nn.Module):
    """只在全连接层 fc1 后加 Dropout"""
    def __init__(self, p=0.5):
        super(CNN_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(p)           # 丢弃概率 0.5
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)             # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)                    # Dropout 只作用于全连接层
        x = self.fc2(x)
        return x


# BN + Dropout 组合
class CNN_BN_Dropout(nn.Module):
    """卷积层加 BatchNorm，全连接层加 Dropout"""
    def __init__(self, p=0.5):
        super(CNN_BN_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(p)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)                        # BN
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)                        # BN
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)             # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)                    # Dropout
        x = self.fc2(x)
        return x

def train_eval(model, train_loader, test_loader, epoch_num=10, lr=0.001):
    """训练并评估，返回损失与准确率历史"""
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loss_list = []
    test_acc_list = []

    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            opt.zero_grad()                    # 梯度清零
            outputs = model(images)
            loss = loss_fn(outputs, labels)    # 计算损失
            loss.backward()                    # 反向传播
            opt.step()                         # 参数更新
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_loss_list.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100 * correct / total
        test_acc_list.append(acc)
        print(f"Epoch {epoch+1}, Train Loss:{avg_loss:.4f}, Test Acc:{acc:.2f}%")

    return train_loss_list, test_acc_list
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
cifar_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
cifar_test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)
# 四组独立实验
experiments = {
    '无正则化'   : CNN(),
    '仅BN'      : CNN_BN(),
    '仅Dropout' : CNN_Dropout(),
    'BN+Dropout': CNN_BN_Dropout(),
}

results = {}
for name, model in experiments.items():
    print(f"\n========== {name} ==========")
    loss_list, acc_list = train_eval(model, cifar_train_loader, cifar_test_loader)
    results[name] = (loss_list, acc_list)



# 2. 数据增强（1课时）
# 使用torchvision.transforms实现：
# o随机水平翻转、随机裁剪、随机旋转
# o色彩抖动、随机灰度化
# 对比有无数据增强的训练效果
# 分析数据增强对泛化能力的提升
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ===== 无数据增强：仅转张量 + 归一化 =====
transform_plain = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ===== 有数据增强：随机翻转 / 裁剪 / 旋转 / 色彩抖动 / 灰度化 =====
transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转，概率 0.5
    transforms.RandomCrop(32, padding=4),        # 随机裁剪（先补边 4 像素再裁回 32×32）
    transforms.RandomRotation(degrees=15),       # 随机旋转，范围 ±15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                                 # 色彩抖动：亮度/对比度/饱和度/色相
    transforms.RandomGrayscale(p=0.1),           # 随机灰度化，概率 0.1
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ===== 分别构建两套数据集与加载器 =====
train_set_plain = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_plain)
test_set        = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_plain)  # 测试集不用增强
train_set_aug   = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_aug)

train_loader_plain = DataLoader(train_set_plain, batch_size=64, shuffle=True)
train_loader_aug   = DataLoader(train_set_aug,   batch_size=64, shuffle=True)
test_loader        = DataLoader(test_set,        batch_size=64, shuffle=False)

# 基准 CNN：无 BatchNorm、无 Dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        x = x.view(-1, 64 * 8 * 8)      # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)      # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_eval(model, train_loader, test_loader, epoch_num=10, lr=0.001):
    """返回训练损失、训练准确率、测试准确率，用于观察泛化/过拟合"""
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loss_list, train_acc_list, test_acc_list = [], [], []

    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        for images, labels in train_loader:
            opt.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_loss_list.append(avg_loss)
        train_acc_list.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100 * correct / total
        test_acc_list.append(acc)
        print(f"Epoch {epoch+1}, Loss:{avg_loss:.4f}, "
              f"Train Acc:{train_acc:.2f}%, Test Acc:{acc:.2f}%")

    return train_loss_list, train_acc_list, test_acc_list


print("========== 无数据增强 ==========")
model_plain = CNN()
loss_plain, train_acc_plain, test_acc_plain = \
    train_eval(model_plain, train_loader_plain, test_loader)

print("\n========== 有数据增强 ==========")
model_aug = CNN()
loss_aug, train_acc_aug, test_acc_aug = \
    train_eval(model_aug, train_loader_aug, test_loader)



# 3. 预训练模型特征提取（1课时）
# 加载预训练ResNet18模型
# 冻结所有卷积层参数
# 替换最后的全连接层为自定义分类头
# 在自定义小数据集（如猫狗分类）上训练
# 对比从头训练与特征提取的效果

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

data_dir = './data/cat_dog'
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize(224),                    # ResNet 输入 224×224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 均值/方差
])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_set = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform_train)
test_set  = datasets.ImageFolder(root=f'{data_dir}/test',  transform=transform_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)
print("类别:", train_set.classes)   # ['cat', 'dog']
print("训练样本数:", len(train_set))
# 加载 ImageNet 预训练 ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print("原始分类头:", resnet.fc)

num_classes = 2
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

for param in resnet.parameters():
    param.requires_grad = False

for param in resnet.fc.parameters():
    param.requires_grad = True

print("新分类头:", resnet.fc)

# ========== 2. 两个模型构建函数 ==========
def build_resnet_from_scratch(num_classes=2):
    """从头训练：不加载预训练权重，全部参数可训练"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_resnet_frozen(num_classes=2):
    """特征提取：预训练 + 冻结卷积层，只训练分类头"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = False          # 冻结卷积层
    for param in model.fc.parameters():
        param.requires_grad = True           # 分类头可训练
    return model



def train_eval(model, train_loader, test_loader, epoch_num=10, lr=0.001):
    """训练并评估，返回损失与准确率历史"""
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD([p for p in model.parameters() if p.requires_grad],  # 只优化可训练参数
                    lr=lr, momentum=0.9)

    train_loss_list, test_acc_list = [], []
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
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100 * correct / total
        test_acc_list.append(acc)
        print(f"Epoch {epoch+1}, Loss:{avg_loss:.4f}, Test Acc:{acc:.2f}%")

    return train_loss_list, test_acc_list

def train_eval(model, train_loader, test_loader, epoch_num=10, lr=0.001):
    """训练并评估，返回损失与准确率历史"""
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD([p for p in model.parameters() if p.requires_grad],  # 只优化可训练参数
                    lr=lr, momentum=0.9)

    train_loss_list, test_acc_list = [], []
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
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100 * correct / total
        test_acc_list.append(acc)
        print(f"Epoch {epoch+1}, Loss:{avg_loss:.4f}, Test Acc:{acc:.2f}%")

    return train_loss_list, test_acc_list


'''
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ========== 1. 数据准备 ==========
data_dir = './data/cat_dog'

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_set = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform_train)
test_set  = datasets.ImageFolder(root=f'{data_dir}/test',  transform=transform_test)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)
print("类别:", train_set.classes)
print("训练样本数:", len(train_set))

# ========== 2. 两个模型构建函数 ==========
def build_resnet_from_scratch(num_classes=2):
    """从头训练：不加载预训练权重，全部参数可训练"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_resnet_frozen(num_classes=2):
    """特征提取：预训练 + 冻结卷积层，只训练分类头"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = False          # 冻结卷积层
    for param in model.fc.parameters():
        param.requires_grad = True           # 分类头可训练
    return model

# ========== 3. 训练函数（只定义一次） ==========
def train_eval(model, train_loader, test_loader, epoch_num=10, lr=0.001):
    """训练并评估，返回损失与准确率历史"""
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD([p for p in model.parameters() if p.requires_grad],
                    lr=lr, momentum=0.9)
    train_loss_list, test_acc_list = [], []
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
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100 * correct / total
        test_acc_list.append(acc)
        print(f"Epoch {epoch+1}, Loss:{avg_loss:.4f}, Test Acc:{acc:.2f}%")
    return train_loss_list, test_acc_list

# ========== 4. 真正训练两个模型（生成 loss_s 等变量） ==========
print("========== 从头训练 ==========")
model_scratch = build_resnet_from_scratch()
loss_s, acc_s = train_eval(model_scratch, train_loader, test_loader)

print("\n========== 特征提取（预训练+冻结） ==========")
model_frozen = build_resnet_frozen()
loss_f, acc_f = train_eval(model_frozen, train_loader, test_loader)

# ========== 5. 画图对比 ==========
epochs = range(1, 11)

plt.figure(figsize=(6, 4))
plt.plot(epochs, loss_s, marker='o', label='从头训练')
plt.plot(epochs, loss_f, marker='o', label='特征提取')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('迁移学习：训练损失对比')
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(6, 4))
plt.plot(epochs, acc_s, marker='o', label='从头训练')
plt.plot(epochs, acc_f, marker='o', label='特征提取')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('迁移学习：测试准确率对比')
plt.legend(); plt.grid(True); plt.show()

print(f"\n从头训练 最终测试准确率：{acc_s[-1]:.2f}%")
print(f"特征提取 最终测试准确率：{acc_f[-1]:.2f}%")

'''


# 4. 迁移学习微调（1课时）
# 解冻ResNet最后两个残差块
# 使用较小学习率进行微调
# 对比三种方式：从头训练 / 特征提取 / 微调
# 记录准确率和收敛速度，分析迁移学习优势







# （四）考核要求
# 提交正则化对比实验代码和结果
# 提交迁移学习完整代码
# 实验报告包含三种方式对比分析