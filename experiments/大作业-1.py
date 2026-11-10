# ===================== 1. 导入依赖库 =====================
import os
# 关闭 oneDNN 警告
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Lambda
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# 设置中文显示（绘图用）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 重写兼容新版TF的注意力层（修复核心报错） =====================
def attention_layer(x):
    # 计算注意力得分
    score = Dense(1, activation='tanh')(x)
    # 在时序维度 axis=2 做softmax
    attn_weight = tf.nn.softmax(score, axis=1)
    # 加权特征
    weighted = x * attn_weight
    # 求和得到全局特征
    output = tf.reduce_sum(weighted, axis=1)
    return output

# ===================== 3. 加载 IMDB 文本数据集 =====================
MAX_VOCAB_SIZE = 5000
MAX_SEQ_LEN = 100
EMBEDDING_DIM = 128

# 英文影评数据集：二分类（正面/负面情感）
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_VOCAB_SIZE)
num_classes = 2

print(f"训练集样本数：{len(x_train)}")
print(f"测试集样本数：{len(x_test)}")
print(f"分类类别数：{num_classes}")

# 标签转为独热编码
y_train_onehot = to_categorical(y_train, num_classes)
y_test_onehot = to_categorical(y_test, num_classes)

# 文本序列补齐
X_train_pad = pad_sequences(x_train, maxlen=MAX_SEQ_LEN)
X_test_pad = pad_sequences(x_test, maxlen=MAX_SEQ_LEN)

print(f"训练集输入形状：{X_train_pad.shape}")
print(f"测试集输入形状：{X_test_pad.shape}")

# ===================== 4. 构建基础LSTM模型（补充input_shape，正常显示参数） =====================
def build_base_lstm():
    model = Sequential()
    model.add(Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_shape=(MAX_SEQ_LEN,)
    ))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))   # 增大dropout缓解过拟合
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===================== 5. 构建 LSTM + 注意力机制 模型 =====================
def build_lstm_attention():
    inputs = Input(shape=(MAX_SEQ_LEN,))
    embed = Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    )(inputs)
    lstm_out = LSTM(64, return_sequences=True)(embed)
    lstm_out = Dropout(0.3)(lstm_out)
    # 调用注意力
    attn_out = Lambda(attention_layer)(lstm_out)
    output = Dense(num_classes, activation='softmax')(attn_out)
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===================== 6. 模型优化（结构+参数优化） =====================
def build_optimized_model():
    inputs = Input(shape=(MAX_SEQ_LEN,))
    embed = Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    )(inputs)
    lstm1 = LSTM(128, return_sequences=True)(embed)
    lstm1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    lstm2 = Dropout(0.3)(lstm2)
    attn_out = Lambda(attention_layer)(lstm2)
    dense = Dense(64, activation='relu')(attn_out)
    dense = Dropout(0.2)(dense)
    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=output)
    from tensorflow.keras.optimizers import Adam
    opt = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===================== 7. 模型训练 =====================
BATCH_SIZE = 32
EPOCHS = 12   # 减少轮数，进一步抑制过拟合

# 训练基础LSTM
base_model = build_base_lstm()
print("\n========== 基础LSTM模型结构 ==========")
base_model.summary()
history_base = base_model.fit(
    X_train_pad, y_train_onehot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=1
)

# 训练LSTM+注意力模型
attn_model = build_lstm_attention()
print("\n========== LSTM+Attention模型结构 ==========")
attn_model.summary()
history_attn = attn_model.fit(
    X_train_pad, y_train_onehot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=1
)

# 训练优化后模型
opt_model = build_optimized_model()
print("\n========== 优化后LSTM+Attention模型结构 ==========")
opt_model.summary()
history_opt = opt_model.fit(
    X_train_pad, y_train_onehot,
    batch_size=64,
    epochs=18,
    validation_split=0.1,
    verbose=1
)

# ===================== 8. 模型在测试集上评估 =====================
print("\n========== 测试集评估结果 ==========")
base_loss, base_acc = base_model.evaluate(X_test_pad, y_test_onehot, verbose=0)
print(f"基础LSTM 测试集准确率: {base_acc:.4f}")

attn_loss, attn_acc = attn_model.evaluate(X_test_pad, y_test_onehot, verbose=0)
print(f"LSTM+注意力机制 测试集准确率: {attn_acc:.4f}")

opt_loss, opt_acc = opt_model.evaluate(X_test_pad, y_test_onehot, verbose=0)
print(f"优化后LSTM+注意力 测试集准确率: {opt_acc:.4f}")

# ===================== 9. 训练过程可视化 =====================
plt.figure(figsize=(15, 5))

# 准确率曲线
plt.subplot(1,2,1)
plt.plot(history_base.history['accuracy'], label='基础LSTM-训练准确率')
plt.plot(history_base.history['val_accuracy'], label='基础LSTM-验证准确率')
plt.plot(history_attn.history['accuracy'], label='LSTM+Attn-训练准确率')
plt.plot(history_attn.history['val_accuracy'], label='LSTM+Attn-验证准确率')
plt.plot(history_opt.history['accuracy'], label='优化模型-训练准确率')
plt.plot(history_opt.history['val_accuracy'], label='优化模型-验证准确率')
plt.title('模型训练准确率变化')
plt.xlabel('迭代轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

# 损失曲线
plt.subplot(1,2,2)
plt.plot(history_base.history['loss'], label='基础LSTM-训练损失')
plt.plot(history_base.history['val_loss'], label='基础LSTM-验证损失')
plt.plot(history_attn.history['loss'], label='LSTM+Attn-训练损失')
plt.plot(history_attn.history['val_loss'], label='LSTM+Attn-验证损失')
plt.plot(history_opt.history['loss'], label='优化模型-训练损失')
plt.plot(history_opt.history['val_loss'], label='优化模型-验证损失')
plt.title('模型训练损失变化')
plt.xlabel('迭代轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()