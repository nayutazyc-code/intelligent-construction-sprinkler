import cv2
import numpy as np
import threading
import time
import csv
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- 基础路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_PATH = "/Users/nayuta/Desktop/data3.mp4"  # 请确认视频路径正确
CSV_FILE = os.path.join(OUTPUT_DIR, "dust_dataset.csv")
COMMAND_FILE = os.path.join(OUTPUT_DIR, "cannon_command.txt")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LATEST_FRAME_PATH = os.path.join(OUTPUT_DIR, "latest_frame.jpg")
PREVIEW_SIZE = (640, 480)
PREVIEW_JPEG_QUALITY = 75
PREVIEW_SAVE_INTERVAL = 3
ANALYSIS_PLOT_FILE = os.path.join(OUTPUT_DIR, "analysis_result.png")

# --- 初始化全局变量 ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit()

current_pm25, current_pm10, current_tsp = 15.0, 25.0, 40.0
is_currently_dusty = False


# --- 1. 模拟传感器物理线程 (核心逻辑) ---
def mock_sensor_thread():
    global current_pm25, current_pm10, current_tsp, is_currently_dusty
    base_pm25 = 15.0

    while True:
        # A. 读取主控指令
        cannon_on = False
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, "r") as f:
                    if f.read().strip() == '1':
                        cannon_on = True
            except:
                pass

        # B. 确定目标浓度 (Target)
        if cannon_on:
            # 【关键】：只要开启喷淋，目标值强制设为安全区 (25-45)
            target_pm25 = np.random.uniform(25, 45)
            alpha = 0.4  # 下降速度快
        elif is_currently_dusty:
            # 有尘源且没开喷淋，目标值飙升
            target_pm25 = np.random.uniform(180, 280)
            alpha = 0.15  # 上升速度中等
        else:
            # 环境干净且没开喷淋，回归基准
            target_pm25 = base_pm25
            alpha = 0.2

        # C. 一阶惯性滤波 (模拟数值平滑变化)
        current_pm25 += (target_pm25 - current_pm25) * alpha

        # D. 联动计算 PM10 和 TSP (符合物理相关性)
        current_pm10 = current_pm25 * 1.6 + np.random.normal(0, 2)
        current_tsp = current_pm25 * 3.2 + np.random.normal(0, 5)

        # E. 叠加微小噪声并进行边界限制
        current_pm25 = np.clip(current_pm25 + np.random.normal(0, 0.5), 5, 600)
        current_pm10 = np.clip(current_pm10, 10, 900)
        current_tsp = np.clip(current_tsp, 20, 1500)

        time.sleep(1)  # 物理引擎每秒计算一次


# 启动模拟线程
threading.Thread(target=mock_sensor_thread, daemon=True).start()


# --- 2. 绘图与分析功能 ---
def save_analysis_plot():
    if not os.path.exists(CSV_FILE): return
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) < 5: return
        plt.figure(figsize=(14, 7))
        plt.plot(df['timestamp'], df['PM2.5'], label='PM2.5', color='blue', alpha=0.7)
        plt.fill_between(df['timestamp'], 0, df['PM2.5'].max(),
                         where=(df['has_dust_source'] == 1),
                         color='red', alpha=0.15, label='Dust Source Detected')
        plt.title("Physical Simulation: PM2.5 Dynamics")
        plt.xlabel("Time (s)")
        plt.ylabel("Concentration")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(ANALYSIS_PLOT_FILE)
        print(f"\n📈 分析图已生成: {ANALYSIS_PLOT_FILE}")
    except Exception as e:
        print(f"生成图表失败: {e}")


# --- 3. 主程序 (视频处理与数据采集) ---
def main():
    global is_currently_dusty, current_pm25, current_pm10, current_tsp

    print("🚀 视频采集与环境模拟器已启动...")

    # 初始化CSV文件
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'PM2.5', 'PM10', 'TSP', 'has_dust_source'])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {VIDEO_PATH}")
        return

    # 确定尘源类别
    dust_source_ids = [k for k, v in model.names.items() if
                       any(x in v.lower() for x in ['dust', 'dumping', 'excavation', 'truck', 'person'])]

    prev_gray = None
    start_time = time.time()
    last_record_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 视频循环播放
                continue

            # --- A. 视觉检测算法 ---
            # 1. YOLO 检测
            results = model(frame, verbose=False)[0]
            yolo_detected = False
            for box in results.boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                if cls_id in dust_source_ids and conf > 0.4:
                    yolo_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 2. HSV 颜色过滤
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 0, 150), (180, 50, 255))
            dust_ratio = cv2.countNonZero(mask) / (mask.size + 1e-6)

            # 3. 运动检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = 0
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion = np.mean(thresh) / 255
            prev_gray = gray

            # 综合逻辑判断
            is_currently_dusty = yolo_detected or (dust_ratio > 0.02 and motion > 0.01)

            # --- B. 数据持久化 (每 0.5 秒记录一次，提升学习频率) ---
            curr_t = time.time()
            if curr_t - last_record_time >= 0.5:
                elapsed = int(curr_t - start_time)
                with open(CSV_FILE, 'a', newline='') as f_append:
                    csv.writer(f_append).writerow([
                        elapsed,
                        round(current_pm25, 2),
                        round(current_pm10, 2),
                        round(current_tsp, 2),
                        int(is_currently_dusty)
                    ])
                last_record_time = curr_t

            # --- C. UI 实时显示 ---
            color = (0, 0, 255) if is_currently_dusty else (0, 255, 0)
            status_txt = "EVENT: DUSTY" if is_currently_dusty else "EVENT: CLEAN"

            cv2.putText(frame, status_txt, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            cv2.putText(frame, f"PM2.5: {current_pm25:.1f}  TSP: {current_tsp:.1f}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 显示调试信息
            debug_info = f"YOLO: {yolo_detected} | Color: {dust_ratio:.3f} | Motion: {motion:.3f}"
            cv2.putText(frame, debug_info, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            display_frame = cv2.resize(frame, PREVIEW_SIZE)
            frame_count += 1
            if frame_count % PREVIEW_SAVE_INTERVAL == 0:
                temp_frame_path = LATEST_FRAME_PATH + ".tmp.jpg"
                cv2.imwrite(temp_frame_path, display_frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY])
                os.replace(temp_frame_path, LATEST_FRAME_PATH)
            cv2.imshow("Smart Site Environment Simulator", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_analysis_plot()


if __name__ == "__main__":
    main()
