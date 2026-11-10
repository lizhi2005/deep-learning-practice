# ===================== IMDB 文本情感分类实验（最终可运行版） =====================
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭烦人的oneDNN警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 关闭TF多余日志
# 关闭 TensorFlow 警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 解决 matplotlib 中文+负号显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 修正拼写错误

# 深度学习库
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ===================== 1. 超参数设置 =====================
max_words = 10000
epochs = 20
batch_size = 512

# ===================== 2. 加载数据集 =====================
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)

# ===================== 3. 文本向量化（BOW 词袋模型） =====================
def vectorize_sequences(sequences, dimension=max_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.array(train_labels).astype("float32")
y_test = np.array(test_labels).astype("float32")

# 划分验证集
x_val = x_train[:5000]
y_val = y_train[:5000]
x_train_part = x_train[5000:]
y_train_part = y_train[5000:]

# ===================== 4. 构建模型 =====================
def build_best_model():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(max_words,)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    return model

# ===================== 5. 编译模型 =====================
model = build_best_model()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===================== 6. 早停法 =====================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ===================== 7. 训练 =====================
history = model.fit(
    x_train_part, y_train_part,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ===================== 8. 测试集评估 =====================
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ 测试集准确率: {test_acc:.4f}")
print(f"✅ 测试集损失: {test_loss:.4f}")

# ===================== 9. 绘制曲线 =====================
history_dict = history.history

plt.figure(figsize=(14, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history_dict["loss"], label="训练损失")
plt.plot(history_dict["val_loss"], label="验证损失")
plt.title("训练 & 验证 损失曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history_dict["accuracy"], label="训练准确率")
plt.plot(history_dict["val_accuracy"], label="验证准确率")
plt.title("训练 & 验证 准确率曲线")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()