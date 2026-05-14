import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    Layer,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import load_config, runtime_paths

CONFIG = load_config()
PATHS = runtime_paths(CONFIG)
OUTPUT_DIR = PATHS["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = PATHS["data_file"]
MODEL_FILE = PATHS["model_file"]
PREDICTION_PLOT_FILE = PATHS["prediction_plot_file"]
PREDICTION_METRICS_FILE = PATHS["prediction_metrics_file"]
PREDICTION_COMPARISON_PLOT_FILE = PATHS["prediction_comparison_plot_file"]
ATTENTION_HEATMAP_FILE = PATHS["attention_heatmap_file"]
PREDICTION_PREPROCESSOR_FILE = PATHS["prediction_preprocessor_file"]

SEQ_LEN = int(os.environ.get("SMART_SITE_SEQ_LEN", "20"))
PREDICT_HORIZON_STEPS = int(os.environ.get("SMART_SITE_PREDICT_HORIZON_STEPS", "10"))
EPOCHS = int(os.environ.get("SMART_SITE_PREDICT_EPOCHS", "80"))
BATCH_SIZE = int(os.environ.get("SMART_SITE_PREDICT_BATCH_SIZE", "32"))

BASE_FEATURES = ["PM2.5", "PM10", "TSP", "has_dust_source"]
OPTIONAL_FEATURE_DEFAULTS = {
    "temperature": 24.0,
    "humidity": 45.0,
    "wind_speed": 1.2,
    "wind_direction": 90.0,
    "spray_level": 0.0,
    "spray_duration": 0.0,
}
FEATURES = BASE_FEATURES + list(OPTIONAL_FEATURE_DEFAULTS)
TARGET_LABELS = ["PM2.5", "PM10", "TSP"]


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
            name="Att_W",
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],), initializer="zeros", trainable=True, name="Att_b"
        )
        self.u = self.add_weight(
            shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True, name="Att_u"
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        score = tf.tensordot(uit, self.u, axes=1)
        score = tf.squeeze(score, -1)
        weights = tf.nn.softmax(score, axis=1)
        weights_expanded = tf.expand_dims(weights, -1)
        context_vector = tf.reduce_sum(weights_expanded * x, axis=1)
        return context_vector, weights


def prepare_dataframe(path):
    df = pd.read_csv(path)
    for column, default in OPTIONAL_FEATURE_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    feature_df = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    physical_limits = {
        "PM2.5": (0, 800),
        "PM10": (0, 1200),
        "TSP": (0, 2000),
        "has_dust_source": (0, 1),
        "temperature": (-30, 60),
        "humidity": (0, 100),
        "wind_speed": (0, 30),
        "wind_direction": (0, 360),
        "spray_level": (0, 2),
        "spray_duration": (0, 86400),
    }
    for column, (low, high) in physical_limits.items():
        feature_df.loc[(feature_df[column] < low) | (feature_df[column] > high), column] = np.nan

    for column in feature_df.columns:
        if column in {"has_dust_source", "spray_level"}:
            continue
        q1 = feature_df[column].quantile(0.25)
        q3 = feature_df[column].quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            low = q1 - 3.0 * iqr
            high = q3 + 3.0 * iqr
            feature_df.loc[(feature_df[column] < low) | (feature_df[column] > high), column] = np.nan

    feature_df = feature_df.interpolate(limit_direction="both").bfill().ffill()
    feature_df["has_dust_source"] = feature_df["has_dust_source"].round().clip(0, 1)
    feature_df["spray_level"] = feature_df["spray_level"].round().clip(0, 2)
    return feature_df


def create_dataset(data, seq_len=SEQ_LEN, horizon_steps=PREDICT_HORIZON_STEPS):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon_steps + 1):
        X.append(data[i:i + seq_len, :])
        future_window = data[i + seq_len:i + seq_len + horizon_steps, 0:3]
        y.append(future_window.reshape(-1))
    return np.array(X), np.array(y)


def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
    return model


def build_lstm(input_shape, output_size):
    inputs = Input(shape=input_shape)
    x = LSTM(96, return_sequences=False)(inputs)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    return compile_model(Model(inputs=inputs, outputs=Dense(output_size)(x))), None


def build_bilstm(input_shape, output_size):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(96, return_sequences=False))(inputs)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    return compile_model(Model(inputs=inputs, outputs=Dense(output_size)(x))), None


def build_attention_lstm(input_shape, output_size):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    context_vector, attn_weights = Attention()(x)
    x = Dense(128, activation="relu")(context_vector)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    model = compile_model(Model(inputs=inputs, outputs=Dense(output_size)(x)))
    attention_model = Model(inputs=inputs, outputs=attn_weights)
    return model, attention_model


def build_cnn_bilstm(input_shape, output_size):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = Bidirectional(LSTM(80, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    x = Dense(96, activation="relu")(x)
    return compile_model(Model(inputs=inputs, outputs=Dense(output_size)(x))), None


def evaluate_predictions(y_true, y_pred, horizon_steps):
    y_true_3d = y_true.reshape(-1, horizon_steps, 3)
    y_pred_3d = y_pred.reshape(-1, horizon_steps, 3)
    rows = []
    for index, label in enumerate(TARGET_LABELS):
        true_values = y_true_3d[:, :, index].reshape(-1)
        pred_values = y_pred_3d[:, :, index].reshape(-1)
        mse = mean_squared_error(true_values, pred_values)
        rows.append({
            "target": label,
            "MAE": mean_absolute_error(true_values, pred_values),
            "RMSE": np.sqrt(mse),
            "R2": r2_score(true_values, pred_values),
        })
    return rows


def plot_main_prediction(y_true, y_pred, horizon_steps, metrics_by_target):
    y_true_3d = y_true.reshape(-1, horizon_steps, 3)
    y_pred_3d = y_pred.reshape(-1, horizon_steps, 3)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    plot_count = min(100, len(y_true_3d))

    for i, label in enumerate(TARGET_LABELS):
        target_metrics = metrics_by_target[label]
        axes[i].plot(y_true_3d[:plot_count, -1, i], label="Actual", color="#1f77b4", linewidth=2)
        axes[i].plot(y_pred_3d[:plot_count, -1, i], label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
        axes[i].set_title(
            f"{label} Horizon Prediction | MAE: {target_metrics['MAE']:.2f} | "
            f"RMSE: {target_metrics['RMSE']:.2f} | R2: {target_metrics['R2']:.2f}",
            fontsize=12,
        )
        axes[i].legend(loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_FILE, dpi=300)
    plt.close()


def plot_model_comparison(metrics_df):
    pm25_metrics = metrics_df[metrics_df["target"] == "PM2.5"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(pm25_metrics["model"], pm25_metrics[metric], color="#2c7fb8")
        ax.set_title(f"PM2.5 {metric}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PREDICTION_COMPARISON_PLOT_FILE, dpi=300)
    plt.close()


def train_and_evaluate_model(name, builder, input_shape, output_size, X_train, y_train, X_test, y_test, scaler_y):
    print(f"\n训练模型: {name}")
    model, attention_model = builder(input_shape, output_size)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1)
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1,
    )
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return model, attention_model, evaluate_predictions(y_test, y_pred, PREDICT_HORIZON_STEPS), y_pred


def main():
    feature_df = prepare_dataframe(CSV_FILE)
    print("清洗后的数据统计特征:\n", feature_df.describe())
    data = feature_df[FEATURES].values

    X, y = create_dataset(data)
    if len(X) < 20:
        raise ValueError("可训练样本过少，请增加采集数据量或降低 SMART_SITE_PREDICT_HORIZON_STEPS。")

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    n_samples, n_steps, n_feats = X_train.shape
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, n_feats)).reshape(n_samples, n_steps, n_feats)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_feats)).reshape(-1, n_steps, n_feats)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    builders = [
        ("LSTM", build_lstm),
        ("BiLSTM", build_bilstm),
        ("Attention-LSTM", build_attention_lstm),
        ("CNN-BiLSTM", build_cnn_bilstm),
    ]

    metrics_rows = []
    main_model = None
    main_attention_model = None
    main_prediction = None
    for name, builder in builders:
        model, attention_model, target_metrics, y_pred = train_and_evaluate_model(
            name,
            builder,
            input_shape=(n_steps, n_feats),
            output_size=PREDICT_HORIZON_STEPS * 3,
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_test=X_test_scaled,
            y_test=y_test,
            scaler_y=scaler_y,
        )
        for row in target_metrics:
            metrics_rows.append({"model": name, **row})
        if name == "Attention-LSTM":
            main_model = model
            main_attention_model = attention_model
            main_prediction = y_pred

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(PREDICTION_METRICS_FILE, index=False)
    print(f"预测模型对比指标已保存: {PREDICTION_METRICS_FILE}")
    plot_model_comparison(metrics_df)
    print(f"预测模型对比图已保存: {PREDICTION_COMPARISON_PLOT_FILE}")

    main_metrics = {
        row["target"]: row for row in metrics_rows if row["model"] == "Attention-LSTM"
    }
    plot_main_prediction(y_test, main_prediction, PREDICT_HORIZON_STEPS, main_metrics)
    print(f"主预测模型评价图已保存: {PREDICTION_PLOT_FILE}")

    main_model.save(MODEL_FILE)
    print(f"训练结束，主预测模型已保存: {MODEL_FILE}")
    with open(PREDICTION_PREPROCESSOR_FILE, "wb") as f:
        pickle.dump({
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "features": FEATURES,
            "seq_len": SEQ_LEN,
            "horizon_steps": PREDICT_HORIZON_STEPS,
        }, f)
    print(f"预测预处理器已保存: {PREDICTION_PREPROCESSOR_FILE}")

    print("正在生成 Attention Heatmap...")
    num_samples_to_plot = min(30, len(X_test_scaled))
    attention_scores = main_attention_model.predict(X_test_scaled[:num_samples_to_plot], verbose=0)

    plt.figure(figsize=(12, 8))
    plt.imshow(attention_scores, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel(f"Time Step (0 = Oldest data, {n_steps - 1} = Most recent data)", fontsize=12)
    plt.ylabel("Test Sample Index", fontsize=12)
    plt.title("Attention Heatmap (Model Focus Over Time)", fontsize=14)
    plt.xticks(np.arange(0, n_steps))
    plt.yticks(np.arange(0, num_samples_to_plot))
    plt.tight_layout()
    plt.savefig(ATTENTION_HEATMAP_FILE, dpi=300)
    plt.close()
    print(f"Attention Heatmap 已保存: {ATTENTION_HEATMAP_FILE}")


if __name__ == "__main__":
    main()
