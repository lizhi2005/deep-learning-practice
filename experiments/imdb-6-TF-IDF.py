# ===================== IMDB TF-IDF 实验（100% 无报错版） =====================
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer

# ===================== 加载数据 =====================
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# ===================== 文本解码 =====================
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_review(seq):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in seq])

train_texts = [decode_review(x) for x in train_data]
test_texts = [decode_review(x) for x in test_data]

# ===================== TF-IDF 向量化 =====================
tfidf = TfidfVectorizer(max_features=10000)
x_train_tfidf = tfidf.fit_transform(train_texts).toarray()
x_test_tfidf = tfidf.transform(test_texts).toarray()

y_train = np.array(train_labels).astype("float32")
y_test = np.array(test_labels).astype("float32")

# ===================== 自动获取输入维度（解决报错核心！）=====================
input_dim = x_train_tfidf.shape[1]

# ===================== 模型（自动匹配维度，永不报错）=====================
model = Sequential([
    Input(shape=(input_dim,)),  # 自动匹配
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# ===================== 训练 =====================
history = model.fit(
    x_train_tfidf[5000:], y_train[5000:],
    validation_data=(x_train_tfidf[:5000], y_train[:5000]),
    epochs=20,
    batch_size=256,
    callbacks=[early_stop]
)

# ===================== 评估 =====================
test_loss, test_acc = model.evaluate(x_test_tfidf, y_test)
print(f"\n✅ TF-IDF 测试准确率: {test_acc:.4f}")
print(f"✅ TF-IDF 测试损失: {test_loss:.4f}")

# ===================== 绘图 =====================
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="训练损失")
plt.plot(history.history["val_loss"], label="验证损失")
plt.title("TF-IDF 损失曲线")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="训练准确率")
plt.plot(history.history["val_accuracy"], label="验证准确率")
plt.title("TF-IDF 准确率曲线")
plt.legend()

plt.show()