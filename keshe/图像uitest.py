import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QWidget, QTextEdit)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# -------------------------- 全局配置 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "./data/garbage"
batch_size = 16
lr = 1e-3
num_epochs = 15
num_classes = 5
classes = ["cardboard", "metal", "paper", "plastic", "white-glass"]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------- 模型构建 --------------------------
def build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

model = build_model(num_classes).to(device)

# -------------------------- 数据加载（训练用） --------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_root, "test"), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"训练集数量: {len(train_dataset)}")
print(f"验证集数量: {len(val_dataset)}")
print(f"测试集数量: {len(test_dataset)}")
print("类别映射：", train_dataset.class_to_idx)

# -------------------------- 训练循环 --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    epoch_train_loss = running_loss / total
    epoch_train_acc = correct / total

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
    epoch_val_loss = val_running_loss / val_total
    epoch_val_acc = val_correct / val_total

    train_loss_list.append(epoch_train_loss)
    train_acc_list.append(epoch_train_acc)
    val_loss_list.append(epoch_val_loss)
    val_acc_list.append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss:{epoch_train_loss:.4f} Acc:{epoch_train_acc:.4f} | Val Loss:{epoch_val_loss:.4f} Acc:{epoch_val_acc:.4f}")

torch.save(model.state_dict(), "garbage_resnet18.pth")
print("模型已保存为 garbage_resnet18.pth")

# 绘制loss、acc曲线
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="train loss")
plt.plot(val_loss_list, label="val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(train_acc_list, label="train acc")
plt.plot(val_acc_list, label="val acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.tight_layout()
plt.savefig("train_curve.png")
plt.show()

# -------------------------- 测试集评估 + 混淆矩阵 --------------------------
model.load_state_dict(torch.load("garbage_resnet18.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, pred = torch.max(outputs, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("===== 分类报告 =====")
print(classification_report(all_labels, all_preds, target_names=classes))

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=30)
plt.yticks(tick_marks, classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j,i,str(cm[i][j]), ha="center", va="center")
plt.xlabel("Predict")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -------------------------- 推理接口 --------------------------
def infer_image(pil_img):
    img = test_transform(pil_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(img)
        prob = torch.softmax(logits, dim=1)[0]
    pred_idx = torch.argmax(prob).item()
    pred_name = classes[pred_idx]
    conf = float(prob[pred_idx])
    return pred_name, conf

# -------------------------- PyQt窗口（支持拖拽+选择图片） --------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("垃圾分类识别系统 PyQt5")
        self.resize(600, 700)
        self.setAcceptDrops(True)  # 开启拖拽

        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.img_label = QLabel("👉 拖拽图片到这里 / 点击下方按钮选择图片")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border:1px solid #999; padding:10px;")
        self.img_label.setFixedHeight(350)

        self.btn_select = QPushButton("选择本地图片")
        self.btn_select.clicked.connect(self.select_file)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        layout.addWidget(self.img_label)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.result_text)
        self.setCentralWidget(widget)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.do_recognize(path)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片(*.jpg *.png *.jpeg *.bmp)")
        if path:
            self.do_recognize(path)

    def do_recognize(self, img_path):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            # 预览
            pix = QPixmap(img_path).scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(pix)
            # 推理
            res, conf = infer_image(pil_img)
            self.result_text.setText(f"识别结果：{res}\n置信度：{conf:.2%}")
        except Exception as e:
            self.result_text.setText(f"识别失败：{str(e)}")

if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
