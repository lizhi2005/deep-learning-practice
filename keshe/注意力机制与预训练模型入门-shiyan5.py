# （一）实验目的
# 1.理解注意力机制的基本原理
# 2.掌握Self-Attention的计算过程
# 3.了解Transformer的基本结构
# 4.能够使用BERT进行文本分类微调
# （二）实验内容
# 序号	内容	课时
# 1	手动实现注意力机制	1
# 2	Self-Attention与Transformer入门	1
# 3	BERT模型加载与微调	1.5
# 4	对比实验与结果分析	0.5
# （三）实现过程
# 1. 注意力机制
# 实现Bahdanau注意力
# 在LSTM情感分析中加入注意力机制
# 可视化注意力权重，分析模型关注哪些词
# 2. Self-Attention
# 实现Scaled Dot-Product Attention
# 实现Multi-Head Attention（简化版，2个头）
# 了解Transformer的Encoder结构
# 使用Transformer Encoder完成文本分类
# 3. BERT微调
# 使用HuggingFace Transformers加载预训练BERT
# 加载中文情感分析数据集
# 在BERT顶部添加分类头
# 微调BERT完成情感分类
# 4. 对比实验
# 对比：LSTM / LSTM+Attention / Transformer / BERT
# 记录准确率、F1值、训练时间
# 分析各模型优缺点
# （四）考核要求
# 提交注意力机制实现代码
# 提交BERT微调代码及结果
# 实验报告包含四种模型对比分析

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------- 数据 ----------------------
# 中文用ChnSentiCorp酒店评论，四个模型都在同一份数据上跑，比才有意义
import jieba
from collections import Counter

max_len = 100
max_words = 8000
batch_size = 64
embed_dim = 128
hidden_size = 64
lr = 1e-3
num_epochs = 4

def clean_text(text):
    # 只留汉字、字母、数字，标点全扔
    return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fa5" or ch.isalnum())

def tokenize(text):
    return [w for w in jieba.lcut(clean_text(text)) if w.strip()]

train_df = pd.read_csv("chnsenticorp/train.csv")
val_df = pd.read_csv("chnsenticorp/validation.csv")
test_df = pd.read_csv("chnsenticorp/test.csv")
print(f"训练{len(train_df)}条，验证{len(val_df)}条，测试{len(test_df)}条")

train_tokens = [tokenize(t) for t in train_df["text"]]
val_tokens = [tokenize(t) for t in val_df["text"]]
test_tokens = [tokenize(t) for t in test_df["text"]]

counter = Counter()
for toks in train_tokens:
    counter.update(toks)
vocab = {"<pad>": 0, "<unk>": 1}
for word, _ in counter.most_common(max_words - 2):
    vocab[word] = len(vocab)
print(f"词表大小: {len(vocab)}")

def encode(tokens):
    ids = [vocab.get(w, 1) for w in tokens[:max_len]]
    return ids + [0] * (max_len - len(ids))

x_train = torch.tensor([encode(t) for t in train_tokens])
y_train = torch.tensor(train_df["label"].values)
x_val = torch.tensor([encode(t) for t in val_tokens])
y_val = torch.tensor(val_df["label"].values)
x_test = torch.tensor([encode(t) for t in test_tokens])
y_test = torch.tensor(test_df["label"].values)

from torch.utils.data import TensorDataset, DataLoader
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 四个模型共用一套训练/评估，超参保持一致
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, epochs=num_epochs, lr=lr):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running += l.item()
        preds, trues = evaluate(model, val_loader)
        print(f"  Epoch {epoch + 1}/{epochs} | loss={running / len(train_loader):.4f} | 验证acc={accuracy_score(trues, preds):.4f}")
    return model

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu())
            trues.append(yb)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


# ---------------------- LSTM + 注意力 ----------------------
# Bahdanau的加性注意力：e_t = v^T tanh(W h_t)，softmax完加权求和
# 必须把<pad>屏蔽掉，不然补的0也会分走一部分权重
class LSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, return_attn=False):
        out, _ = self.lstm(self.embedding(x))        # (B, T, H)
        e = self.v(torch.tanh(self.W(out))).squeeze(-1)
        e = e.masked_fill(x == 0, -1e9)
        alpha = torch.softmax(e, dim=1)
        context = (alpha.unsqueeze(-1) * out).sum(dim=1)
        logits = self.fc(context)
        if return_attn:
            return logits, alpha
        return logits

print("\nLSTM+注意力")
m1 = LSTMAttentionClassifier(len(vocab), embed_dim, hidden_size).to(device)
t0 = time.time()
m1 = train_model(m1)
t_attn = time.time() - t0
preds, trues = evaluate(m1, test_loader)
acc1, f1_1 = accuracy_score(trues, preds), f1_score(trues, preds)
print(f"测试 acc={acc1:.4f} | F1={f1_1:.4f} | 用时{t_attn:.0f}s")

# 随便挑一条测试评论，看权重压在哪些词上
idx = 8
words = test_tokens[idx][:max_len]
_, alpha = m1(x_test[idx].unsqueeze(0).to(device), return_attn=True)
weights = alpha[0].detach().cpu().numpy()[:len(words)]

plt.figure(figsize=(12, 4))
plt.bar(range(len(words)), weights)
plt.xticks(range(len(words)), words, rotation=60, fontsize=9)
plt.title(f"注意力权重（真实标签：{'正面' if y_test[idx] == 1 else '负面'}）")
plt.ylabel("注意力权重")
plt.tight_layout()
plt.show()


# ---------------------- Self-Attention ----------------------
# softmax(QK^T/√d_k)·V，除根号d是为了点积别太大把softmax推进饱和区
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return attn @ V, attn

# 多头就2个，多了CPU上跑不动
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=2):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)

mh = MultiHeadAttention(embed_dim, num_heads=2)
print("\n多头注意力形状检查: (2,10,128) ->", mh(torch.randn(2, 10, embed_dim)).shape)


# ---------------------- Transformer Encoder ----------------------
# Embedding+可学习位置编码 -> EncoderLayer x2 -> 带mask的平均池化 -> 全连接
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        emb = self.embedding(x) + self.pos[:, :x.size(1)]
        out = self.encoder(emb)
        mask = (x != 0).unsqueeze(-1).float()        # pad位置不参与池化
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(pooled)

print("\nTransformer Encoder")
m2 = TransformerClassifier(len(vocab), embed_dim, num_heads=4, num_layers=2).to(device)
t0 = time.time()
m2 = train_model(m2)
t_tf = time.time() - t0
preds2, trues2 = evaluate(m2, test_loader)
acc2, f1_2 = accuracy_score(trues2, preds2), f1_score(trues2, preds2)
print(f"测试 acc={acc2:.4f} | F1={f1_2:.4f} | 用时{t_tf:.0f}s")


# ---------------------- BERT微调 ----------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

bert_path = r"D:\深度学习综合设计\dl_course_design\models\bert-base-chinese"
# 本地没有就换成 "bert-base-chinese" 让它自己下
tok = AutoTokenizer.from_pretrained(bert_path)
bert = AutoModelForSequenceClassification.from_pretrained(bert_path, num_labels=2).to(device)

# BERT有自己的字级tokenizer，前面jieba那套词表用不上
bert_maxlen = 128
bert_batch = 16
bert_epochs = 2
bert_lr = 2e-5      # 微调只能用很小的学习率，大了直接把预训练权重冲坏

def bert_encode(texts):
    return tok(list(texts), truncation=True, padding="max_length",
               max_length=bert_maxlen, return_tensors="pt")

bert_train = bert_encode(train_df["text"])
bert_val = bert_encode(val_df["text"])
bert_test = bert_encode(test_df["text"])
bert_train_loader = DataLoader(TensorDataset(bert_train["input_ids"], bert_train["attention_mask"],
                                             torch.tensor(y_train)), batch_size=bert_batch, shuffle=True)
bert_val_loader = DataLoader(TensorDataset(bert_val["input_ids"], bert_val["attention_mask"],
                                           torch.tensor(y_val)), batch_size=32)
bert_test_loader = DataLoader(TensorDataset(bert_test["input_ids"], bert_test["attention_mask"],
                                            torch.tensor(y_test)), batch_size=32)

print("\nBERT微调，CPU上大概半小时，先干别的去")
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(bert.parameters(), lr=bert_lr)
t0 = time.time()
for epoch in range(bert_epochs):
    bert.train()
    running = 0.0
    for ids, mask, yb in bert_train_loader:
        ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
        opt.zero_grad()
        out = bert(input_ids=ids, attention_mask=mask)
        l = loss_fn(out.logits, yb)
        l.backward()
        opt.step()
        running += l.item()
    bert.eval()
    vp, vt = [], []
    with torch.no_grad():
        for ids, mask, yb in bert_val_loader:
            out = bert(input_ids=ids.to(device), attention_mask=mask.to(device))
            vp.append(out.logits.argmax(dim=1).cpu())
            vt.append(yb)
    print(f"  Epoch {epoch + 1}/{bert_epochs} | loss={running / len(bert_train_loader):.4f} | 验证acc={accuracy_score(torch.cat(vt).numpy(), torch.cat(vp).numpy()):.4f}")
t_bert = time.time() - t0

tp, tt = [], []
with torch.no_grad():
    for ids, mask, yb in bert_test_loader:
        out = bert(input_ids=ids.to(device), attention_mask=mask.to(device))
        tp.append(out.logits.argmax(dim=1).cpu())
        tt.append(yb)
bert_preds = torch.cat(tp).numpy()
bert_trues = torch.cat(tt).numpy()
acc3, f1_3 = accuracy_score(bert_trues, bert_preds), f1_score(bert_trues, bert_preds)
print(f"测试 acc={acc3:.4f} | F1={f1_3:.4f} | 用时{t_bert:.0f}s")


# 前面先写了三个，LSTM基线最后补上，凑齐四个再比
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, (h, c) = self.lstm(self.embedding(x))
        return self.fc(h[-1])

print("\nLSTM基线")
m0 = LSTMClassifier(len(vocab), embed_dim, hidden_size).to(device)
t0 = time.time()
m0 = train_model(m0)
t_lstm = time.time() - t0
preds0, trues0 = evaluate(m0, test_loader)
acc0, f1_0 = accuracy_score(trues0, preds0), f1_score(trues0, preds0)
print(f"测试 acc={acc0:.4f} | F1={f1_0:.4f} | 用时{t_lstm:.0f}s")


# 四个模型摆一起看
print("\n" + "=" * 50)
print("四种模型对比（ChnSentiCorp中文情感）")
print("=" * 50)
names = ["LSTM", "LSTM+Attention", "Transformer", "BERT"]
accs = [acc0, acc1, acc2, acc3]
f1s = [f1_0, f1_1, f1_2, f1_3]
times = [t_lstm, t_attn, t_tf, t_bert]
for i, name in enumerate(names):
    print(f"{name:16s} | acc={accs[i]:.4f} | F1={f1s[i]:.4f} | 用时{times[i]:.0f}s")

# 柱状图，acc和F1并排
x_pos = np.arange(len(names))
plt.figure(figsize=(10, 5))
w = 0.35
plt.bar(x_pos - w / 2, accs, w, label="准确率")
plt.bar(x_pos + w / 2, f1s, w, label="F1值")
plt.xticks(x_pos, names)
plt.ylim(0.5, 1.0)
plt.title("四种模型在中文情感分类上的对比")
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
