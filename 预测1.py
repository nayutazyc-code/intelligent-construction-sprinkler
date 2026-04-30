import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(OUTPUT_DIR, "dust_dataset.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "dust_attention_lstm_model.keras")
PREDICTION_PLOT_FILE = os.path.join(OUTPUT_DIR, "optimized_prediction.png")
ATTENTION_HEATMAP_FILE = os.path.join(OUTPUT_DIR, "attention_heatmap.png")


# 1. Attention 层定义
# 升级版的 Attention 层 (Bahdanau 风格)
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (batch_size, time_steps, features)
        # 1. 权重矩阵 W，用于转换输入特征
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True, name="Att_W")
        # 2. 偏置 b
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True, name="Att_b")
        # 3. 核心改进：上下文向量 u，用于把 64 维特征压缩成 1 个单一的分数！
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True, name="Att_u")
        super(Attention, self).build(input_shape)

    def call(self, x):
        # x 形状: (batch, 20, 64)
        # 第一步：特征非线性变换
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)

        # 第二步：将 64 维特征压缩成 1 个标量分数
        # 形状变为: (batch, 20, 1)
        score = tf.tensordot(uit, self.u, axes=1)

        # 把多余的维度挤掉，形状变为: (batch, 20)
        score = tf.squeeze(score, -1)

        # 第三步：计算概率权重 (加起来等于 1)
        weights = tf.nn.softmax(score, axis=1)

        # 第四步：加权求和
        # 扩展 weights 形状使其能与 x 相乘: (batch, 20, 1)
        weights_expanded = tf.expand_dims(weights, -1)
        context_vector = tf.reduce_sum(weights_expanded * x, axis=1)

        return context_vector, weights

# 2. 数据处理
df = pd.read_csv(CSV_FILE)
print("数据统计特征:\n", df.describe())

features = ['PM2.5', 'PM10', 'TSP', 'has_dust_source']
data = df[features].interpolate().bfill().ffill().values


def create_dataset(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len, :])
        y.append(data[i + seq_len, 0:3])
    return np.array(X), np.array(y)


X, y = create_dataset(data, seq_len=20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

n_samples, n_steps, n_feats = X_train.shape
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, n_feats)).reshape(n_samples, n_steps, n_feats)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_feats)).reshape(-1, n_steps, n_feats)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 3. 构建增强版模型
input_layer = Input(shape=(n_steps, n_feats))

lstm_out = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
lstm_out = Dropout(0.2)(lstm_out)
lstm_out = LSTM(64, return_sequences=True)(lstm_out)

# --- 修改点 1：捕获 Attention 权重 ---
# 以前是 context_vector, _ = ... 现在把权重赋值给 attn_weights
context_vector, attn_weights = Attention()(lstm_out)

x = Dense(128, activation='relu')(context_vector)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(3)(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.1,
    epochs=300,
    batch_size=32,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# ==========================================
# 5. 预测与评估绘图
# ==========================================
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = y_test

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
labels = ['PM2.5', 'PM10', 'TSP']

for i in range(3):
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mse) # 计算 RMSE
    r2 = r2_score(y_true[:, i], y_pred[:, i])

    axes[i].plot(y_true[:100, i], label='Actual', color='#1f77b4', linewidth=2)
    axes[i].plot(y_pred[:100, i], label='Predicted', color='#ff7f0e', linestyle='--', linewidth=2)

    axes[i].set_title(f"{labels[i]} Prediction | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}", fontsize=12)
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(PREDICTION_PLOT_FILE, dpi=300)
plt.show()

model.save(MODEL_FILE)
print(f"训练结束，主预测模型已保存: {MODEL_FILE}")


# ==========================================
# 6. 生成 Attention Heatmap (热力图)
# ==========================================
print("⏳ 正在生成 Attention Heatmap...")

# 构建专门输出权重的子模型
attention_model = Model(inputs=input_layer, outputs=attn_weights)

# 选取测试集的前 30 个样本
num_samples_to_plot = min(30, len(X_test_scaled))
sample_inputs = X_test_scaled[:num_samples_to_plot]

# 直接预测出二维的注意力权重，形状为 (30, 20)
attention_scores = attention_model.predict(sample_inputs)

plt.figure(figsize=(12, 8))
# cmap='viridis' 是深色到亮黄色的渐变，适合展示权重高低
plt.imshow(attention_scores, cmap='viridis', aspect='auto')

plt.colorbar(label='Attention Weight (Probability)')
plt.xlabel('Time Step (0 = Oldest data, 19 = Most recent data)', fontsize=12)
plt.ylabel('Test Sample Index', fontsize=12)
plt.title('Attention Heatmap (Model Focus Over Time)', fontsize=14)

plt.xticks(np.arange(0, n_steps))
plt.yticks(np.arange(0, num_samples_to_plot))

plt.tight_layout()
plt.savefig(ATTENTION_HEATMAP_FILE, dpi=300)
plt.show()

print(f"Attention Heatmap 生成完毕并已保存为: {ATTENTION_HEATMAP_FILE}")
