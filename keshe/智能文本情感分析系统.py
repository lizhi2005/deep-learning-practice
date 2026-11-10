import pandas as pd
import jieba
import re
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 全局配置 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 100
max_words = 8000
batch_size = 64
embed_dim = 128
hidden_size = 64
lr = 1e-3
num_epochs = 4
bert_path = r"D:\深度学习综合设计\dl_course_design\models\bert-base-chinese"
bert_maxlen = 128
bert_batch = 16
bert_epochs = 2
bert_lr = 2e-5

# -------------------------- 数据预处理 --------------------------
def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.strip()

def tokenize(text):
    return [w for w in jieba.lcut(clean_text(text)) if w.strip()]

train_df = pd.read_csv("chnsenticorp/train.csv").dropna(subset=["text", "label"])
val_df = pd.read_csv("chnsenticorp/validation.csv").dropna(subset=["text", "label"])
test_df = pd.read_csv("chnsenticorp/test.csv").dropna(subset=["text", "label"])
print(f"训练集：{len(train_df)}条 | 验证集：{len(val_df)}条 | 测试集：{len(test_df)}条")

train_tokens = [tokenize(t) for t in train_df["text"]]
val_tokens = [tokenize(t) for t in val_df["text"]]
test_tokens = [tokenize(t) for t in test_df["text"]]

from collections import Counter
counter = Counter()
for toks in train_tokens:
    counter.update(toks)
vocab = {"<pad>": 0, "<unk>": 1}
for word, _ in counter.most_common(max_words - 2):
    vocab[word] = len(vocab)

def encode(tokens):
    # 超长截断，不足补0
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

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

# -------------------------- LSTM基线 --------------------------
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # 按真实长度打包，pad不进LSTM，h[-1]才是评论真正的末尾状态
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.embedding(x), lengths,
                                      batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed)
        return self.fc(h[-1])

def predict_ids(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu())
            trues.append(yb)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

# 训练一次要十几分钟，训完存下来，第二次跑直接加载
if os.path.exists("lstm_sentiment.pth"):
    model_lstm = LSTMClassifier(len(vocab), embed_dim, hidden_size).to(device)
    model_lstm.load_state_dict(torch.load("lstm_sentiment.pth", map_location=device))
    print("已加载保存好的LSTM模型")
else:
    model_lstm = LSTMClassifier(len(vocab), embed_dim, hidden_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model_lstm.parameters(), lr=lr)
    print("开始训练LSTM...")
    for epoch in range(num_epochs):
        model_lstm.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = loss_fn(model_lstm(xb), yb)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=5.0)
            opt.step()
            running += l.item()
        preds, trues = predict_ids(model_lstm, val_loader)
        print(f"  Epoch {epoch + 1}/{num_epochs} | loss={running / len(train_loader):.4f} | 验证acc={accuracy_score(trues, preds):.4f}")
    torch.save(model_lstm.state_dict(), "lstm_sentiment.pth")

lstm_preds, lstm_trues = predict_ids(model_lstm, test_loader)
lstm_acc = accuracy_score(lstm_trues, lstm_preds)
lstm_f1 = f1_score(lstm_trues, lstm_preds)
print(f"LSTM测试集: acc={lstm_acc:.4f} | F1={lstm_f1:.4f}")


# -------------------------- BERT微调 --------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained(bert_path)

def bert_encode(texts):
    return tok(list(texts), truncation=True, padding="max_length",
               max_length=bert_maxlen, return_tensors="pt")

if os.path.exists("bert_sentiment/config.json"):
    model_bert = AutoModelForSequenceClassification.from_pretrained("bert_sentiment").to(device)
    print("已加载保存好的BERT模型")
else:
    model_bert = AutoModelForSequenceClassification.from_pretrained(bert_path, num_labels=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model_bert.parameters(), lr=bert_lr)
    bt_train = bert_encode(train_df["text"])
    bt_val = bert_encode(val_df["text"])
    bt_test = bert_encode(test_df["text"])
    btrain_loader = DataLoader(TensorDataset(bt_train["input_ids"], bt_train["attention_mask"],
                                            torch.tensor(y_train)), batch_size=bert_batch, shuffle=True)
    bval_loader = DataLoader(TensorDataset(bt_val["input_ids"], bt_val["attention_mask"],
                                          torch.tensor(y_val)), batch_size=32)
    print("开始微调BERT...")
    for epoch in range(bert_epochs):
        model_bert.train()
        running = 0.0
        for ids, mask, yb in btrain_loader:
            ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
            opt.zero_grad()
            out = model_bert(input_ids=ids, attention_mask=mask)
            l = loss_fn(out.logits, yb)
            l.backward()
            opt.step()
            running += l.item()
        model_bert.eval()
        vp, vt = [], []
        with torch.no_grad():
            for ids, mask, yb in bval_loader:
                out = model_bert(input_ids=ids.to(device), attention_mask=mask.to(device))
                vp.append(out.logits.argmax(dim=1).cpu())
                vt.append(yb)
        print(f"  Epoch {epoch + 1}/{bert_epochs} | loss={running / len(btrain_loader):.4f} | 验证acc={accuracy_score(torch.cat(vt).numpy(), torch.cat(vp).numpy()):.4f}")
    model_bert.save_pretrained("bert_sentiment")
    tok.save_pretrained("bert_sentiment")
    # 测试集的tensor也存下来，省得下面和界面里再tokenize一遍
    torch.save({"test_ids": bt_test["input_ids"], "test_mask": bt_test["attention_mask"],
                "test_y": torch.tensor(y_test)}, "bert_test_cache.pt")

cache = torch.load("bert_test_cache.pt")
btest_loader = DataLoader(TensorDataset(cache["test_ids"], cache["test_mask"],
                                       cache["test_y"]), batch_size=32)
bp, bt = [], []
model_bert.eval()
with torch.no_grad():
    for ids, mask, yb in btest_loader:
        out = model_bert(input_ids=ids.to(device), attention_mask=mask.to(device))
        bp.append(out.logits.argmax(dim=1).cpu())
        bt.append(yb)
bert_preds = torch.cat(bp).numpy()
bert_trues = torch.cat(bt).numpy()
bert_acc = accuracy_score(bert_trues, bert_preds)
bert_f1 = f1_score(bert_trues, bert_preds)
print(f"BERT测试集: acc={bert_acc:.4f} | F1={bert_f1:.4f}")


# -------------------------- 结果对比 --------------------------
print("\n" + "=" * 50)
print("两个模型对比（测试集）")
print("=" * 50)
print(f"LSTM       | acc={lstm_acc:.4f} | P={precision_score(lstm_trues, lstm_preds):.4f} "
      f"| R={recall_score(lstm_trues, lstm_preds):.4f} | F1={lstm_f1:.4f}")
print(f"BERT(微调) | acc={bert_acc:.4f} | P={precision_score(bert_trues, bert_preds):.4f} "
      f"| R={recall_score(bert_trues, bert_preds):.4f} | F1={bert_f1:.4f}")
print("=" * 50)
print("\nLSTM分类报告：")
print(classification_report(lstm_trues, lstm_preds, target_names=["负面", "正面"]))
print("BERT分类报告：")
print(classification_report(bert_trues, bert_preds, target_names=["负面", "正面"]))


# -------------------------- PyQt5交互界面 --------------------------
# 输入框 + 预测按钮 + 结果区，LSTM和BERT各判一次，两个结果一起显示
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QWidget, QTextEdit)


class SentimentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能文本情感分析系统")
        self.resize(560, 380)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("在这里输入要分析的中文评论...")

        self.btn = QPushButton("预测情感")
        self.btn.clicked.connect(self.on_predict)

        self.result_label = QLabel("等待输入...")
        self.result_label.setStyleSheet("font-size: 15px;")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("输入评论："))
        layout.addWidget(self.input_edit)
        layout.addWidget(self.btn)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_predict(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            self.result_label.setText("请先输入评论内容")
            return
        # LSTM这条走jieba分词再编码
        ids = torch.tensor([encode(tokenize(text))]).to(device)
        model_lstm.eval()
        with torch.no_grad():
            p1 = torch.softmax(model_lstm(ids), dim=1)[0]
        l_label = "正面" if p1.argmax().item() == 1 else "负面"

        # BERT用自己那套tokenizer
        enc = tok([text], truncation=True, padding="max_length",
                  max_length=bert_maxlen, return_tensors="pt")
        model_bert.eval()
        with torch.no_grad():
            p2 = torch.softmax(model_bert(input_ids=enc["input_ids"].to(device),
                                          attention_mask=enc["attention_mask"].to(device)).logits, dim=1)[0]
        b_label = "正面" if p2.argmax().item() == 1 else "负面"

        self.result_label.setText(
            f"LSTM判断：{l_label}（置信度 {p1.max().item():.2%}）\n"
            f"BERT判断：{b_label}（置信度 {p2.max().item():.2%}）"
        )


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = SentimentApp()
    win.show()
    sys.exit(app.exec_())
