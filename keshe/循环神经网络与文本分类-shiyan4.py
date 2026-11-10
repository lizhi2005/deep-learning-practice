# （一）实验目的
# 1.理解RNN的基本原理和链式结构
# 2.掌握LSTM的门控机制
# 3.掌握文本预处理和词嵌入方法
# 4.能够构建RNN模型完成文本分类
# （二）实验内容
# 序号	内容	课时
# 1	手动实现RNN前向传播	1
# 2	LSTM原理与实现	1
# 3	文本预处理与词嵌入	1
# 4	LSTM情感分析实战	1
# （三）实现过程
# 1. 手动实现RNN（1课时）
# 实现RNN单元：h_t = tanh(Wxh·x_t + Whh·h_{t-1} + b)
# 实现序列前向传播
# 演示梯度消失问题（长序列梯度趋近于0）
# 使用梯度裁剪缓解梯度爆炸
# 2. LSTM实现（1课时）
# 手动实现LSTM的四个门
# 对比RNN和LSTM在长序列上的表现
# 3. 文本预处理（1课时）
# 加载IMDB电影评论数据集
# 文本清洗：去除HTML标签、特殊字符
# 分词与构建词表（Vocabulary）
# 序列填充/截断到固定长度
# 实现词嵌入层（Embedding）
# 4. 情感分析实战（1课时）
# 构建LSTM分类模型
# 训练模型，记录损失和准确率
# 测试集评估，计算准确率和F1值
# 对新评论进行情感预测
# （四）考核要求
# 提交手动实现的RNN/LSTM代码
# 提交情感分析完整代码及结果
# 实验报告包含梯度问题分析

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


# 权重自己用nn.Parameter建，不走nn.RNN，写一遍才知道里面在算什么
# h_t = tanh(x_t·Wxh + h_{t-1}·Whh + b)
class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wxh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.5)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.5)
        self.bh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        batch, T, _ = x.shape          # batch, 时间步, 特征
        h = torch.zeros(batch, self.Whh.size(0))    # 初始隐状态全0
        for t in range(T):
            h = torch.tanh(x[:, t] @ self.Wxh.T + h @ self.Whh.T + self.bh)
        return h                       # 只要最后时刻的

rnn = ManualRNN(8, 16)
out_check = rnn(torch.randn(2, 5, 8))
print("===== 手动RNN前向传播 =====")
print("输出形状:", out_check.shape)


# loss只是最后时刻h的函数，梯度却要穿过T个tanh才能回到第1步
# T一大，早期时间步的梯度就没了，这就是梯度消失
def grad_norm_per_step(model, T):
    x = torch.randn(1, T, 8, requires_grad=True)
    loss = (model(x) ** 2).sum()
    loss.backward()
    return [x.grad[:, t, :].norm().item() for t in range(T)]


print("\n不同序列长度下，第1步的梯度范数：")
grad_record = {}
for T in [10, 40, 80]:
    m = ManualRNN(8, 16)
    norms = grad_norm_per_step(m, T)
    grad_record[T] = norms
    print(f"T={T:3d} | 第1步={norms[0]:.3e} | 中间步={norms[T // 2]:.3e} | 最后一步={norms[-1]:.3e}")

# 画出来直观点，三条线从右往左塌
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 5))
for i, T in enumerate([10, 40, 80]):
    plt.subplot(1, 3, i + 1)
    plt.plot(range(1, T + 1), grad_record[T], marker='o', markersize=3)
    plt.title(f"序列长度 T={T}")
    plt.xlabel("时间步 t")
    plt.ylabel("梯度范数")
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 梯度爆炸：loss乘100把梯度撑起来，再用clip_grad_norm_压回1
print("\n===== 梯度裁剪 =====")
m = ManualRNN(8, 16)
x = torch.randn(1, 80, 8, requires_grad=True)
((m(x) ** 2).sum() * 100).backward()
before = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
print(f"裁剪前参数梯度总范数: {before:.2f}")
torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
after = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
print(f"裁剪后参数梯度总范数: {after:.2f}")


# LSTM四个门，[h,x]直接concat，等价于拆成两个矩阵再相加
# f=σ(Wf[h,x]) 遗忘门   i=σ(Wi[h,x]) 输入门
# C=f*C_prev + i*tanh(Wc[h,x])   o=σ(Wo[h,x])   h=o*tanh(C)
class ManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        # 遗忘门偏置要置个正数，不然随机初始化下f≈0.5，C通路每步衰减一半，
        # 长序列上LSTM衰减得比RNN还快（Greff 2017那篇里提过这个trick）
        nn.init.constant_(self.Wf.bias, 4.0)

    def forward(self, x):
        batch, T, _ = x.shape
        h = torch.zeros(batch, self.hidden_size)
        c = torch.zeros(batch, self.hidden_size)
        for t in range(T):
            comb = torch.cat([x[:, t], h], dim=1)
            f = torch.sigmoid(self.Wf(comb))
            i = torch.sigmoid(self.Wi(comb))
            c_hat = torch.tanh(self.Wc(comb))
            o = torch.sigmoid(self.Wo(comb))
            c = f * c + i * c_hat
            h = o * torch.tanh(c)
        return h


# RNN和LSTM比一下第1步剩多少梯度，LSTM有f*C_prev这条近线性的路，衰减慢
print("\n===== RNN 与 LSTM 长序列对比 =====")
for T in [10, 40, 80]:
    g_rnn = grad_norm_per_step(ManualRNN(8, 16), T)[0]
    m_lstm = ManualLSTM(8, 16)
    x = torch.randn(1, T, 8, requires_grad=True)
    (m_lstm(x) ** 2).sum().backward()
    print(f"T={T:3d} | RNN={g_rnn:.3e} | LSTM={x.grad[:, 0, :].norm().item():.3e}")


# ---------------------- IMDB 预处理 ----------------------
import pandas as pd
import re
from collections import Counter

max_words = 10000      # 词表只留最高频的1万个，再大也学不动
max_len = 200          # 评论统一到200个词
batch_size = 64
embed_dim = 128
hidden_size = 64
num_layers = 2
num_epochs = 3
lr = 1e-3

def clean_text(text):
    # IMDB里<br />这类标签不少，先整个去掉，再只留字母
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r"[^a-zA-Z']", ' ', text)
    return text.lower()

def tokenize(text):
    return clean_text(text).split()

train_df = pd.read_csv("imdb/train.csv")
test_df = pd.read_csv("imdb/test.csv")
print(f"\n训练集{len(train_df)}条，测试集{len(test_df)}条")
print(train_df["label"].value_counts())

train_tokens = [tokenize(t) for t in train_df["text"]]
test_tokens = [tokenize(t) for t in test_df["text"]]
print("分词样例:", train_tokens[0][:10])

# 0留给<pad>，1留给<unk>，剩下的按词频排
counter = Counter()
for toks in train_tokens:
    counter.update(toks)
vocab = {"<pad>": 0, "<unk>": 1}
for word, _ in counter.most_common(max_words - 2):
    vocab[word] = len(vocab)
print(f"词表大小: {len(vocab)}")

def encode(tokens):
    # 超长截断，不够后面补0
    ids = [vocab.get(w, 1) for w in tokens[:max_len]]
    return ids + [0] * (max_len - len(ids))

x_train = torch.tensor([encode(t) for t in train_tokens])
y_train = torch.tensor(train_df["label"].values)
x_test = torch.tensor([encode(t) for t in test_tokens])
y_test = torch.tensor(test_df["label"].values)
print(f"编码后形状: 训练{x_train.shape} 测试{x_test.shape}")


from torch.utils.data import TensorDataset, DataLoader

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

from torch.nn.utils.rnn import pack_padded_sequence

# Embedding 10000->128，LSTM 128->64两层，最后接64->2
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # 按真实长度打包，pad那段不进LSTM。一开始没打包，200长度大半是pad，
        # h[-1]被pad稀释得只剩66%准确率，打包后才正常
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.embedding(x), lengths,
                                      batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed)
        return self.fc(h[-1])       # 取最后一层最后一个时刻

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(len(vocab), embed_dim, hidden_size, num_layers).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)
print(model)

# 这台机器只有CPU，3轮大概十分钟，再多就没意义了
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        l = loss_fn(model(xb), yb)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)   # 训练LSTM一般都裁一下，防爆
        opt.step()
        running += l.item()
    avg_loss = running / len(train_loader)
    loss_list.append(avg_loss)

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(dim=1)
            all_preds.append(preds.cpu())
            correct += (preds == yb.to(device)).sum().item()
            total += yb.size(0)
    acc = correct / total
    acc_list.append(acc)
    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")


from sklearn.metrics import f1_score, classification_report

all_preds = torch.cat(all_preds).numpy()
print(f"\n测试集准确率 {correct / total:.4f}，F1 {f1_score(y_test.numpy(), all_preds):.4f}")
print(classification_report(y_test.numpy(), all_preds, target_names=["负面", "正面"]))

# 损失和准确率画一块
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_list, marker='o')
plt.title("训练损失变化曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), acc_list, marker='o', color='orange')
plt.title("测试集准确率变化曲线")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 拿几条没见过的评论试一下
def predict_sentiment(text):
    model.eval()
    ids = torch.tensor([encode(tokenize(text))]).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(ids), dim=1)[0]
    label = prob.argmax().item()
    return ("正面" if label == 1 else "负面"), prob[label].item()


print("\n===== 新评论预测 =====")
reviews = [
    "This movie is wonderful, the story is moving and the actors are great.",
    "Terrible film, waste of time and money, the plot is boring.",
    "One of the best movies I have ever seen, highly recommended."
]
for r in reviews:
    label, p = predict_sentiment(r)
    print(f"[{label}] 置信度={p:.4f} | {r[:50]}...")

torch.save(model.state_dict(), "lstm_imdb.pth")
print("模型已保存: lstm_imdb.pth")
