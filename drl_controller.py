from collections import deque
import json
import random
import matplotlib.pyplot as plt
import os, sys, time, subprocess, pandas as pd, numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(OUTPUT_DIR, "dust_dataset.csv")
TRAINER_SCRIPT = os.path.join(BASE_DIR, "预测1.py")
MODEL_FILE = os.path.join(OUTPUT_DIR, "dust_attention_lstm_model.keras")
COLLECTOR_SCRIPT = os.path.join(BASE_DIR, "collector.py")
COMMAND_FILE = os.path.join(OUTPUT_DIR, "cannon_command.txt")
EVALUATION_PLOT_FILE = os.path.join(OUTPUT_DIR, "drl_multi_metrics_evaluation.png")
STATUS_FILE = os.path.join(OUTPUT_DIR, "system_status.json")

MIN_DATA_ROWS = 800
SEQ_LEN = 20
STATE_SIZE = 5
ACTION_SIZE = 2
BATCH_SIZE = 32
PM25_SAFE_THRESHOLD = 75.0
TSP_SAFE_THRESHOLD = 200.0
MIN_SWITCH_INTERVAL = 60


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
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay


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
    scaled[0][3] /= 300.0  # Predicted PM2.5
    return scaled

def main():
    print("=" * 60)
    print("智慧工地控制系统")
    print("=" * 60)

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
            return

    if not os.path.exists(MODEL_FILE):
        print(f"\n\n[2] 数据达标！启动 {TRAINER_SCRIPT} 训练预测大脑...")
        write_status("training", "数据采集完成，正在训练 Attention-LSTM 预测模型。", rows=get_data_length())
        trainer_proc = subprocess.Popen([sys.executable, TRAINER_SCRIPT], cwd=BASE_DIR)
        trainer_proc.wait()
        if trainer_proc.returncode != 0:
            write_status("error", "Attention-LSTM 模型训练失败，请检查训练脚本输出。", rows=get_data_length())
            return
    else:
        write_status("training", "已检测到训练好的 Attention-LSTM 模型，正在加载模型。", rows=get_data_length())

    print("\n[3] 唤醒 DQN 决策引擎...")
    write_status("control", "模型已就绪，正在进入 DRL 喷淋控制阶段。", rows=get_data_length())
    lstm_model = load_model(MODEL_FILE, custom_objects={'Attention': Attention})
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    df = pd.read_csv(DATA_FILE)
    features = ['PM2.5', 'PM10', 'TSP', 'has_dust_source']
    data = df[features].apply(pd.to_numeric, errors='coerce').interpolate().bfill().ffill().values

    # 修复 Bug 2: 严格按照 attention-lstm1.py 的方式拟合 StandardScaler
    scaler_X = StandardScaler().fit(data)
    scaler_y = StandardScaler().fit(data[:, 0:3])

    current_cannon_status = 0
    last_row_count = get_data_length()
    prev_state, prev_action = None, None
    last_switch_time = 0

    record_pm25 = []
    record_tsp = []
    record_action = []

    print("\n (按 Ctrl+C 停止并生成报告)\n" + "-" * 60)

    try:
        while True:
            current_rows = get_data_length()
            if current_rows > last_row_count:
                last_row_count = current_rows

                # --- 1. 获取当前观测数据 ---
                df_latest = pd.read_csv(DATA_FILE).tail(SEQ_LEN)
                latest_data = df_latest[features].apply(pd.to_numeric,
                                                        errors='coerce').interpolate().bfill().ffill().values
                actual_pm25 = latest_data[-1, 0]
                actual_pm10 = latest_data[-1, 1]
                actual_tsp = latest_data[-1, 2]

                # LSTM 预测
                latest_data_scaled = scaler_X.transform(latest_data).reshape(1, SEQ_LEN, len(features))
                pred_scaled = lstm_model.predict(latest_data_scaled, verbose=0)
                predicted_pm25 = scaler_y.inverse_transform(pred_scaled)[0][0]

                # 构建当前状态并归一化
                current_state_raw = np.array([[actual_pm25, actual_pm10, actual_tsp, predicted_pm25, current_cannon_status]])
                current_state = scale_state(current_state_raw)

                # --- 2. 计算【上一秒动作】的奖励 ---
                if prev_state is not None:
                    reward = 0
                    is_safe = (actual_pm25 <= PM25_SAFE_THRESHOLD) and (actual_tsp <= TSP_SAFE_THRESHOLD)
                    # A. 达标判定
                    if is_safe:
                        reward += 10.0  # 基础生存奖

                        # 省水
                        if prev_action == 0:
                            if actual_pm25 < (PM25_SAFE_THRESHOLD * 0.6):
                                reward += 25.0
                            else:
                                reward += 10.0
                        else:
                            reward -= 15.0  # 安全还开水，重罚！
                    else:
                        #  超标惩罚
                        penalty_pm = (actual_pm25 - PM25_SAFE_THRESHOLD) * 5.0
                        penalty_tsp = (actual_tsp - TSP_SAFE_THRESHOLD) * 2.0  # TSP 权重稍微调低，因为数值基数大
                        reward = - max(penalty_pm, penalty_tsp)

                        # 补偿性奖励
                        if prev_action == 1:
                            reward += 5.0

                            # E. 预判性奖励
                    prev_actual = prev_state[0][0] * 300.0
                    if predicted_pm25 > PM25_SAFE_THRESHOLD and prev_action == 1:
                        reward += 10.0  # 奖励它看预测行事

                    agent.remember(prev_state, prev_action, reward, current_state, False)
                    agent.replay()

                    # --- 4. 做出决策 ---
                action = agent.act(current_state)

                now = time.time()
                if action != current_cannon_status:
                    if (now - last_switch_time) >= MIN_SWITCH_INTERVAL:
                        # 满足 1 分钟间隔，允许切换
                        final_action = action
                        last_switch_time = now
                        # print(f" [保护机制] 满足间隔，允许切换为: {final_action}")
                    else:
                        # 不满足间隔，强制维持原状
                        final_action = current_cannon_status
                        # print(f" [保护机制] 间隔不足，拦截切换请求")
                else:
                    # 动作没变，维持原状
                    final_action = current_cannon_status

                # 【新增】动作切换惩罚逻辑 (放在这里更准确)
                # 如果当前动作和上一秒动作不一样，为了设备寿命，扣一点分
                if prev_action is not None and action != prev_action:
                    # 只有在记忆库里反映这一秒的代价
                    # 也可以直接加在上面的 reward 里
                    pass

                # 写入指令
                with open(COMMAND_FILE, "w") as f:
                    f.write(str(final_action))

                    # 记录数据
                    record_pm25.append(actual_pm25)
                    record_tsp.append(actual_tsp)
                    record_action.append(final_action)

                    # 打印增强日志
                    status_text = "🟢 开" if final_action == 1 else "⚪ 关"
                    print(
                        f"PM2.5: {actual_pm25:.1f} | TSP: {actual_tsp:.1f} | 预测PM2.5: {predicted_pm25:.1f} | 决策: {status_text} | ε: {agent.epsilon:.2f}")

                    current_cannon_status = final_action
                    prev_state = current_state
                    prev_action = final_action

                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n⚠️ 正在生成多指标评价报告...")
        write_status("stopped", "系统已停止，正在生成运行评价报告。", rows=get_data_length())
        collector_proc.kill()
        with open(COMMAND_FILE, "w") as f:
            f.write("0")

        plt.figure(figsize=(14, 8))
        time_axis = range(len(record_pm25))

        # 子图 1: PM2.5 曲线
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, record_pm25, color='#e74c3c', label='Actual PM2.5')
        plt.axhline(y=PM25_SAFE_THRESHOLD, color='green', linestyle='--', label='Threshold')
        plt.fill_between(time_axis, 0, max(record_pm25) + 10, where=(np.array(record_action) == 1),
                                 color='#3498db', alpha=0.2, label='Spraying')
        plt.ylabel("PM2.5"), plt.legend(), plt.grid(True, alpha=0.3)

        # 子图 2: TSP 曲线
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, record_tsp, color='#8e44ad', label='Actual TSP')
        plt.axhline(y=TSP_SAFE_THRESHOLD, color='orange', linestyle='--', label='TSP Limit')
        plt.fill_between(time_axis, 0, max(record_tsp) + 10, where=(np.array(record_action) == 1),
                                 color='#3498db', alpha=0.2)
        plt.ylabel("TSP"), plt.xlabel("Time Steps"), plt.legend(), plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(EVALUATION_PLOT_FILE, dpi=300)
        print(f"✅ 评价图已生成: {EVALUATION_PLOT_FILE}")

if __name__ == "__main__":
    main()
