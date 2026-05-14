from collections import deque
import argparse
import json
import pickle
import random
import matplotlib.pyplot as plt
import os, sys, time, subprocess, pandas as pd, numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler

from config import BASE_DIR, archive_existing_runtime_files, load_config, runtime_paths
from dqn_reward_policy import calculate_reward, classify_pollution

CONFIG = load_config()
PATHS = runtime_paths(CONFIG)
OUTPUT_DIR = PATHS["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = PATHS["data_file"]
MODEL_FILE = os.environ.get("SMART_SITE_DQN_MODEL_FILE") or os.path.join(BASE_DIR, "dust_attention_lstm_model.keras")
DQN_POLICY_MODEL_FILE = (
    os.environ.get("SMART_SITE_DQN_POLICY_MODEL_FILE") or os.path.join(BASE_DIR, "dqn_policy_model.keras")
)
COLLECTOR_SCRIPT = os.path.join(BASE_DIR, "collector.py")
COMMAND_FILE = PATHS["command_file"]
EVALUATION_PLOT_FILE = PATHS["evaluation_plot_file"]
DQN_CONTROL_LOG_FILE = PATHS["dqn_control_log_file"]
STATUS_FILE = PATHS["status_file"]
PREDICTION_PREPROCESSOR_FILE = PATHS.get("prediction_preprocessor_file")

MIN_DATA_ROWS = CONFIG["min_data_rows"]
SEQ_LEN = 20
STATE_SIZE = 8
ACTION_SIZE = 3
BATCH_SIZE = 32
PM25_SAFE_THRESHOLD = 75.0
TSP_SAFE_THRESHOLD = 200.0
MIN_SWITCH_INTERVAL = 60
DEFAULT_MAX_CONTROL_STEPS = 5000
OPTIONAL_FEATURE_DEFAULTS = {
    "temperature": 24.0,
    "humidity": 45.0,
    "wind_speed": 1.2,
    "wind_direction": 90.0,
    "spray_level": 0.0,
    "spray_duration": 0.0,
}


def write_status(stage, message, rows=None):
    payload = {
        "stage": stage,
        "message": message,
        "updated_at": time.time(),
    }
    if rows is not None:
        payload["rows"] = rows

    temp_path = STATUS_FILE + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(temp_path, STATUS_FILE)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DQN spray control experiment.")
    parser.add_argument("--max-control-steps", type=int, default=DEFAULT_MAX_CONTROL_STEPS,
                        help="Control steps to run after entering the DQN control stage.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for repeatable experiments.")
    return parser.parse_args()


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["SMART_SITE_SEED"] = str(seed)


def save_control_log(control_log):
    if not control_log:
        print("暂无 DQN 控制日志，未生成 CSV。")
        return
    pd.DataFrame(control_log).to_csv(DQN_CONTROL_LOG_FILE, index=False)
    print(f"DQN 控制日志已生成: {DQN_CONTROL_LOG_FILE}")


def save_policy_model(agent, reason):
    agent.model.save(DQN_POLICY_MODEL_FILE)
    print(f"DQN 策略模型已保存: {DQN_POLICY_MODEL_FILE} ({reason})")


def save_evaluation_plot(record_pm25, record_tsp, record_action):
    if not record_pm25 or not record_tsp or not record_action:
        print("暂无 DRL 控制记录，未生成评价图。")
        return

    plt.figure(figsize=(14, 8))
    time_axis = range(len(record_pm25))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, record_pm25, color='#e74c3c', label='Actual PM2.5')
    plt.axhline(y=PM25_SAFE_THRESHOLD, color='green', linestyle='--', label='Threshold')
    plt.fill_between(time_axis, 0, max(record_pm25) + 10, where=(np.array(record_action) == 1),
                     color='#3498db', alpha=0.2, label='Spraying')
    plt.ylabel("PM2.5"), plt.legend(), plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, record_tsp, color='#8e44ad', label='Actual TSP')
    plt.axhline(y=TSP_SAFE_THRESHOLD, color='orange', linestyle='--', label='TSP Limit')
    plt.fill_between(time_axis, 0, max(record_tsp) + 10, where=(np.array(record_action) == 1),
                     color='#3498db', alpha=0.2)
    plt.ylabel("TSP"), plt.xlabel("Time Steps"), plt.legend(), plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EVALUATION_PLOT_FILE, dpi=300)
    plt.close()
    print(f"评价图已生成: {EVALUATION_PLOT_FILE}")


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True, name="Att_W")
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True, name="Att_b")
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True, name="Att_u")
        super(Attention, self).build(input_shape)

    def call(self, x):
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        score = tf.tensordot(uit, self.u, axes=1)
        score = tf.squeeze(score, -1)
        weights = tf.nn.softmax(score, axis=1)
        weights_expanded = tf.expand_dims(weights, -1)
        context_vector = tf.reduce_sum(weights_expanded * x, axis=1)
        return context_vector, weights

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE: return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done: target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def get_data_length():
    if not os.path.exists(DATA_FILE): return 0
    try:
        df = pd.read_csv(DATA_FILE)
        return len(df)
    except: return 0

def scale_state(state):
    scaled = state.astype(float).copy()
    scaled[0][0] /= 300.0  # PM2.5
    scaled[0][1] /= 300.0  # PM10
    scaled[0][2] /= 800.0  # TSP
    scaled[0][3] /= 300.0  # Predicted PM2.5 peak
    scaled[0][4] /= 300.0  # Predicted PM2.5 average
    scaled[0][5] /= 300.0  # Pollution trend
    scaled[0][6] /= 2.0    # Spray level
    scaled[0][7] /= 300.0  # Spray duration
    return scaled


def ensure_feature_columns(df, features):
    prepared = df.copy()
    for column, default in OPTIONAL_FEATURE_DEFAULTS.items():
        if column not in prepared.columns:
            prepared[column] = default
    return prepared[features].apply(pd.to_numeric, errors="coerce").interpolate().bfill().ffill()


def load_prediction_preprocessor(data):
    fallback_features = ["PM2.5", "PM10", "TSP", "has_dust_source"]
    if PREDICTION_PREPROCESSOR_FILE and os.path.exists(PREDICTION_PREPROCESSOR_FILE):
        try:
            with open(PREDICTION_PREPROCESSOR_FILE, "rb") as f:
                payload = pickle.load(f)
            return {
                "scaler_X": payload["scaler_X"],
                "scaler_y": payload["scaler_y"],
                "features": payload.get("features", fallback_features),
                "seq_len": int(payload.get("seq_len", SEQ_LEN)),
                "horizon_steps": int(payload.get("horizon_steps", 1)),
                "source": "saved_preprocessor",
            }
        except Exception as exc:
            print(f"预测预处理器读取失败，将使用运行数据重建: {exc}")

    features = fallback_features
    scaler_X = RobustScaler().fit(data[features].apply(pd.to_numeric, errors="coerce").interpolate().bfill().ffill().values)
    scaler_y = RobustScaler().fit(data[["PM2.5", "PM10", "TSP"]].apply(pd.to_numeric, errors="coerce").interpolate().bfill().ffill().values)
    return {
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "features": features,
        "seq_len": SEQ_LEN,
        "horizon_steps": 1,
        "source": "runtime_fallback",
    }


def inverse_prediction(pred_scaled, scaler_y, horizon_steps):
    pred_scaled = np.asarray(pred_scaled)
    output_size = pred_scaled.shape[-1]
    expected_size = horizon_steps * 3
    if output_size == expected_size:
        pred_flat = scaler_y.inverse_transform(pred_scaled)[0]
        return pred_flat.reshape(horizon_steps, 3)
    if output_size == 3:
        return scaler_y.inverse_transform(pred_scaled)[0].reshape(1, 3)
    clipped = pred_scaled[:, :3]
    return scaler_y.inverse_transform(clipped)[0].reshape(1, 3)


def policy_model_compatible(model):
    return (
        model.input_shape[-1] == STATE_SIZE
        and model.output_shape[-1] == ACTION_SIZE
    )

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("智慧工地控制系统")
    print("=" * 60)

    if os.environ.get("SMART_SITE_RUNTIME_PREPARED") != "1":
        archive_dir, moved_files = archive_existing_runtime_files(CONFIG)
        if moved_files:
            print(f"已归档历史运行文件: {archive_dir}")
            print(f"本次控制流程将从空数据状态开始，共归档 {len(moved_files)} 个文件。")

    print("\n[1] 启动虚拟物理环境 (collector.py)...")
    write_status("collecting", f"正在采集初始数据: 0/{MIN_DATA_ROWS}", rows=0)
    collector_proc = subprocess.Popen([sys.executable, COLLECTOR_SCRIPT], cwd=BASE_DIR)

    while get_data_length() < MIN_DATA_ROWS:
        rows = get_data_length()
        write_status("collecting", f"正在采集初始数据: {rows}/{MIN_DATA_ROWS}", rows=rows)
        print(f"\r等待初始数据积累: {rows}/{MIN_DATA_ROWS} 行...", end="")
        time.sleep(2)
        # 防闪退提示：如果 collector.py 出错了，这里会立刻停止并报错
        if collector_proc.poll() is not None:
            write_status("error", "视频采集程序意外退出，请检查视频路径。")
            print("\n 错误：视频采集程序(collector.py)意外闪退！请检查视频路径是否正确。")
            sys.exit(1)

    if not os.path.exists(MODEL_FILE):
        message = f"未找到根目录预测模型: {MODEL_FILE}"
        write_status("error", message, rows=get_data_length())
        print(f"\n错误：{message}")
        sys.exit(1)

    print(f"\n\n[2] 使用根目录 Attention-LSTM 模型: {MODEL_FILE}")
    write_status("training", "已检测到根目录 Attention-LSTM 模型，正在加载模型。", rows=get_data_length())

    print("\n[3] 唤醒 DQN 决策引擎...")
    write_status("control", "模型已就绪，正在进入 DRL 喷淋控制阶段。", rows=get_data_length())
    lstm_model = load_model(MODEL_FILE, custom_objects={'Attention': Attention}, compile=False)
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    policy_loaded = False
    if os.path.exists(DQN_POLICY_MODEL_FILE):
        loaded_policy = load_model(DQN_POLICY_MODEL_FILE, compile=False)
        if policy_model_compatible(loaded_policy):
            agent.model = loaded_policy
            agent.epsilon = agent.epsilon_min
            policy_loaded = True
            print(f"已加载 DQN 策略模型: {DQN_POLICY_MODEL_FILE}")
            print("DQN 将以固定策略推理模式运行，不再执行 replay 在线训练。")
        else:
            print("检测到旧版 DQN 策略模型与三档喷淋状态空间不兼容，将重新在线学习。")
    else:
        print(f"未找到 DQN 策略模型: {DQN_POLICY_MODEL_FILE}")
        print("DQN 将在线学习，ε 衰减到 0.01 后自动停止并保存策略模型。")

    df = pd.read_csv(DATA_FILE)
    preprocessor = load_prediction_preprocessor(df)
    features = preprocessor["features"]
    seq_len = preprocessor["seq_len"]
    horizon_steps = preprocessor["horizon_steps"]
    scaler_X = preprocessor["scaler_X"]
    scaler_y = preprocessor["scaler_y"]
    print(f"预测输入特征: {features}")
    print(f"预测窗口: 最近 {seq_len} 步 -> 未来 {horizon_steps} 步 ({preprocessor['source']})")

    current_cannon_status = 0
    last_row_count = get_data_length()
    last_data_seen_time = time.time()
    prev_state, prev_action = None, None
    previous_pm25, previous_tsp = None, None
    last_switch_time = 0

    record_pm25 = []
    record_tsp = []
    record_action = []
    control_log = []
    control_steps = 0

    print(f"\n (控制步数达到 {args.max_control_steps} 或按 Ctrl+C 停止并生成报告)\n" + "-" * 60)

    try:
        while True:
            current_rows = get_data_length()
            if current_rows > last_row_count:
                last_row_count = current_rows
                last_data_seen_time = time.time()

                # --- 1. 获取当前观测数据 ---
                df_latest = pd.read_csv(DATA_FILE).tail(seq_len)
                latest_features = ensure_feature_columns(df_latest, features)
                latest_data = latest_features.values
                latest_row = df_latest.iloc[-1]
                actual_pm25 = latest_data[-1, 0]
                actual_pm10 = latest_data[-1, 1]
                actual_tsp = latest_data[-1, 2]
                has_dust_source = int(latest_features["has_dust_source"].iloc[-1])
                spray_duration = float(latest_features["spray_duration"].iloc[-1]) if "spray_duration" in latest_features else 0.0

                # LSTM 预测
                latest_data_scaled = scaler_X.transform(latest_data).reshape(1, seq_len, len(features))
                pred_scaled = lstm_model.predict(latest_data_scaled, verbose=0)
                pred_window = inverse_prediction(pred_scaled, scaler_y, horizon_steps)
                predicted_pm25 = float(pred_window[-1, 0])
                predicted_pm25_peak = float(np.max(pred_window[:, 0]))
                predicted_pm25_avg = float(np.mean(pred_window[:, 0]))

                # 构建当前状态并归一化
                pollution_trend = 0.0
                if previous_pm25 is not None:
                    pollution_trend = float(actual_pm25 - previous_pm25)
                current_state_raw = np.array([[
                    actual_pm25,
                    actual_pm10,
                    actual_tsp,
                    predicted_pm25_peak,
                    predicted_pm25_avg,
                    pollution_trend,
                    current_cannon_status,
                    spray_duration,
                ]])
                current_state = scale_state(current_state_raw)

                reward_info = {
                    "reward": np.nan,
                    "pollution_state": classify_pollution(actual_pm25, actual_tsp, predicted_pm25_peak),
                    "action_reason": "initial_observation",
                    "pm25_excess": max(actual_pm25 - PM25_SAFE_THRESHOLD, 0.0),
                    "tsp_excess": max(actual_tsp - TSP_SAFE_THRESHOLD, 0.0),
                    "pollution_trend": 0.0,
                    "water_cost": 0.0,
                    "switch_penalty_applied": 0,
                }
                if prev_state is not None:
                    reward_info = calculate_reward(
                        prev_action,
                        actual_pm25,
                        actual_tsp,
                        predicted_pm25_peak,
                        previous_pm25,
                        previous_tsp,
                        previous_action=current_cannon_status,
                    )
                    reward = reward_info["reward"]
                    if not policy_loaded:
                        agent.remember(prev_state, prev_action, reward, current_state, False)
                        agent.replay()

                action = agent.act(current_state)

                now = time.time()
                switched = False
                if action != current_cannon_status:
                    if (now - last_switch_time) >= MIN_SWITCH_INTERVAL:
                        final_action = action
                        last_switch_time = now
                        switched = True
                        # print(f" [保护机制] 满足间隔，允许切换为: {final_action}")
                    else:
                        final_action = current_cannon_status
                        # print(f" [保护机制] 间隔不足，拦截切换请求")
                else:
                    final_action = current_cannon_status

                if prev_action is not None and action != prev_action:
                    pass

                with open(COMMAND_FILE, "w") as f:
                    f.write(str(final_action))

                    record_pm25.append(actual_pm25)
                    record_tsp.append(actual_tsp)
                    record_action.append(final_action)
                    control_steps += 1
                    control_log.append({
                        "step": control_steps,
                        "timestamp": latest_row.get("timestamp", ""),
                        "PM2.5": actual_pm25,
                        "PM10": actual_pm10,
                        "TSP": actual_tsp,
                        "has_dust_source": has_dust_source,
                        "raw_action": action,
                        "final_action": final_action,
                        "spray_level": final_action,
                        "switched": int(switched),
                        "predicted_pm25": predicted_pm25,
                        "predicted_pm25_peak": predicted_pm25_peak,
                        "predicted_pm25_avg": predicted_pm25_avg,
                        "prediction_horizon_steps": horizon_steps,
                        "spray_duration": spray_duration,
                        "epsilon": agent.epsilon,
                        "policy_mode": "inference" if policy_loaded else "online_training",
                        "reward": reward_info["reward"],
                        "pollution_state": reward_info["pollution_state"],
                        "action_reason": reward_info["action_reason"],
                        "pm25_excess": reward_info["pm25_excess"],
                        "tsp_excess": reward_info["tsp_excess"],
                        "pollution_trend": reward_info["pollution_trend"],
                        "water_cost": reward_info["water_cost"],
                        "switch_penalty_applied": reward_info["switch_penalty_applied"],
                    })

                    status_text = ["关", "低档", "高档"][final_action]
                    print(
                        f"[{control_steps}/{args.max_control_steps}] PM2.5: {actual_pm25:.1f} | "
                        f"TSP: {actual_tsp:.1f} | 预测峰值PM2.5: {predicted_pm25_peak:.1f} | "
                        f"决策: {status_text} | ε: {agent.epsilon:.2f}")

                    current_cannon_status = final_action
                    prev_state = current_state
                    prev_action = final_action
                    previous_pm25 = actual_pm25
                    previous_tsp = actual_tsp

                    if control_steps >= args.max_control_steps:
                        print(f"\n已达到最大控制步数 {args.max_control_steps}，准备生成报告。")
                        break

                    if not policy_loaded and agent.epsilon <= agent.epsilon_min:
                        print(f"\nε 已衰减到 {agent.epsilon:.2f}，停止 DQN 在线训练并保存策略模型。")
                        save_policy_model(agent, "epsilon_min_reached")
                        break

                time.sleep(0.5)
            elif time.time() - last_data_seen_time > 15:
                write_status("error", "传感器数据超过 15 秒未更新，已触发安全停机。", rows=current_rows)
                print("\n错误：传感器数据超过 15 秒未更新，已触发安全停机。")
                break
            else:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n⚠️ 正在停止 DRL 控制系统...")
    finally:
        write_status("stopped", "系统已停止，正在生成运行评价报告。", rows=get_data_length())
        if collector_proc.poll() is None:
            collector_proc.kill()
        with open(COMMAND_FILE, "w") as f:
            f.write("0")
        save_control_log(control_log)
        save_evaluation_plot(record_pm25, record_tsp, record_action)

if __name__ == "__main__":
    main()
