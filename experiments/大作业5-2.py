import os
import numpy as np
import jieba
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Bidirectional, LSTM, Dropout, Dense, Concatenate, GaussianNoise
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Model

# ====================== 屏蔽TF冗余日志 ======================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ====================== 自定义带掩码全局最大池化 ======================
class MaskedGlobalMaxPool1D(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            return tf.reduce_max(inputs, axis=1)
        mask_expand = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        masked_x = inputs * mask_expand + (1.0 - mask_expand) * -1e9
        return tf.reduce_max(masked_x, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

# ====================== 自定义掩码注意力层 ======================
class MaskedAttention(Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.w = Dense(units, activation="tanh")
        self.v = Dense(1)

    def call(self, inputs, mask=None):
        score = self.v(self.w(inputs))
        score = tf.squeeze(score, axis=-1)

        if mask is not None:
            mask_float = tf.cast(mask, tf.float32)
            score = score + (1.0 - mask_float) * -1e9

        attn_weight = tf.nn.softmax(score)
        attn_weight_exp = tf.expand_dims(attn_weight, axis=-1)
        output = tf.reduce_sum(inputs * attn_weight_exp, axis=1)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

# ====================== 超参数 ======================
VOCAB_SIZE = 12000
EMBED_DIM = 256
MAX_SEQ_LEN = 200
LSTM_UNIT = 128
CLASS_NUM = 10
BATCH_SIZE = 256
EPOCHS = 40
INIT_LR = 1e-4

# ====================== 数据加载函数 ======================
def load_data():
    print("Building prefix dict from the default dictionary ...")
    jieba.initialize()
    print("Loading model cost 0.6s, Prefix dict built successfully.")
    print("正在加载训练集...")
    # --------------------------
    # 此处替换成你真实读取文本、分词、填充截断代码
    x_train = np.random.randint(1, VOCAB_SIZE, (50000, MAX_SEQ_LEN))
    y_train = np.random.randint(0, CLASS_NUM, (50000,))
    x_val = np.random.randint(1, VOCAB_SIZE, (5000, MAX_SEQ_LEN))
    y_val = np.random.randint(0, CLASS_NUM, (5000,))
    x_test = np.random.randint(1, VOCAB_SIZE, (10000, MAX_SEQ_LEN))
    y_test = np.random.randint(0, CLASS_NUM, (10000,))
    # --------------------------
    print("正在加载验证集...\n正在加载测试集...")
    print(f"训练集样本数：{len(x_train)}")
    print(f"验证集样本数：{len(x_val)}")
    print(f"测试集样本数：{len(x_test)}")
    category = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
    print(f"类别列表：{category}")
    print(f"训练集形状：{x_train.shape}，分类数：{CLASS_NUM}")
    return x_train, y_train, x_val, y_val, x_test, y_test

# ====================== 搭建模型（修复BiLSTM输入错误） ======================
def build_model():
    print("\n========== 三层轻量化BiLSTM+Mask注意力+带掩码全局池化融合模型 ==========")
    inputs = Input(shape=(MAX_SEQ_LEN,), name="input_layer")

    # Embedding开启mask_zero，自动生成并向下传递padding掩码
    emb = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True, name="embedding")(inputs)
    emb = GaussianNoise(stddev=0.08)(emb)
    emb = Dropout(0.3)(emb)

    # 三层BiLSTM：只传入特征，mask由上层Embedding自动传递，不要手动传mask！
    lstm1 = Bidirectional(LSTM(LSTM_UNIT, return_sequences=True))(emb)
    lstm1 = Dropout(0.3)(lstm1)

    lstm2 = Bidirectional(LSTM(LSTM_UNIT, return_sequences=True))(lstm1)
    lstm2 = Dropout(0.3)(lstm2)

    lstm3 = Bidirectional(LSTM(LSTM_UNIT, return_sequences=True))(lstm2)
    lstm3 = Dropout(0.3)(lstm3)

    # 注意力、池化层会自动接收上游传下来的mask，无需外部传入
    attn_out = MaskedAttention()(lstm3)
    pool_out = MaskedGlobalMaxPool1D()(lstm3)

    concat = Concatenate()([attn_out, pool_out])

    dense1 = Dense(512, activation="relu")(concat)
    dense1 = Dropout(0.4)(dense1)
    dense2 = Dense(256, activation="relu")(dense1)
    dense2 = Dropout(0.4)(dense2)
    output = Dense(CLASS_NUM, activation="softmax")(dense2)

    model = Model(inputs=inputs, outputs=output)
    model.summary()
    return model

# ====================== 学习率衰减策略 ======================
def lr_decay_schedule(epoch):
    decay_rate = 0.94
    lr = INIT_LR * (decay_rate ** epoch)
    print(f"\nEpoch {epoch+1}: LearningRateScheduler set lr = {lr:.8f}")
    return lr

# ====================== 训练入口 ======================
if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    model = build_model()

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        LearningRateScheduler(lr_decay_schedule, verbose=1),
        ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max"
        )
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 加载最优模型评估测试集
    print("\n===== 使用最优模型评估测试集 =====")
    best_model = tf.keras.models.load_model(
        "best_model.keras",
        custom_objects={
            "MaskedAttention": MaskedAttention,
            "MaskedGlobalMaxPool1D": MaskedGlobalMaxPool1D
        }
    )
    test_loss, test_acc = best_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print(f"测试集 Loss: {test_loss:.4f} | 测试集 Acc: {test_acc:.4f}")