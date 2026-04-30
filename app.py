import os
import subprocess
import sys
import time
import webbrowser

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit.runtime.scriptrunner import get_script_run_ctx

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "dust_dataset.csv")
COMMAND_FILE = os.path.join(BASE_DIR, "cannon_command.txt")
DRL_SCRIPT = os.path.join(BASE_DIR, "drl_controller.py")
MIN_DATA_ROWS = 800
APP_PORT = 8502
APP_URL = f"http://127.0.0.1:{APP_PORT}"


def get_data_length():
    if not os.path.exists(DATA_FILE):
        return 0
    try:
        return len(pd.read_csv(DATA_FILE))
    except Exception:
        return 0


def launch_system():
    print("=" * 60)
    print("智慧工地 AI 抑尘系统启动器")
    print("=" * 60)
    print("\n[1] 启动 DRL 控制系统...")

    drl_proc = subprocess.Popen([sys.executable, DRL_SCRIPT], cwd=BASE_DIR)

    print("[2] 启动网页监控面板...")
    subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        __file__,
        "--server.port",
        str(APP_PORT),
        "--server.address",
        "127.0.0.1",
    ], cwd=BASE_DIR)
    time.sleep(2)
    webbrowser.open(APP_URL)
    print(f"[3] 网页已打开: {APP_URL}")
    print(f"[4] 页面将实时显示数据采集进度，达到 {MIN_DATA_ROWS} 行后进入控制阶段。")
    print("DRL 控制系统仍在运行。按 Ctrl+C 可停止启动器。")

    try:
        drl_proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止 DRL 控制系统...")
        drl_proc.terminate()


def render_dashboard():
    st.set_page_config(page_title="智慧工地 AI 抑尘系统", layout="wide")

    st.title("🏗️ 智慧工地：基于 DRL + LSTM 的抑尘自主控制云平台")

    # --- 侧边栏：状态显示 ---
    st.sidebar.header("系统状态")
    phase_placeholder = st.sidebar.empty()
    progress_placeholder = st.sidebar.empty()
    status_placeholder = st.sidebar.empty()
    pm25_metric = st.sidebar.empty()
    pm10_metric = st.sidebar.empty()
    tsp_metric = st.sidebar.empty()

    # --- 主界面布局 ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 实时 AI 监控画面")
        video_placeholder = st.empty()

    with col2:
        st.subheader("📈 指数实时动态")
        chart_placeholder = st.empty()

    stage_placeholder = st.empty()

    # --- 实时刷新逻辑 ---
    def get_data():
        if os.path.exists(DATA_FILE):
            try:
                df = pd.read_csv(DATA_FILE)
                return df
            except Exception:
                return None
        return None

    def get_command():
        if os.path.exists(COMMAND_FILE):
            with open(COMMAND_FILE, "r") as f:
                return f.read().strip()
        return "0"

    while True:
        df = get_data()
        cmd = get_command()
        rows = 0 if df is None else len(df)
        progress_value = min(rows / MIN_DATA_ROWS, 1.0)

        if rows < MIN_DATA_ROWS:
            phase_placeholder.markdown("### 当前阶段: **数据采集中**")
            progress_placeholder.progress(progress_value)
            status_placeholder.markdown(f"### 正在采集数据: **{rows}/{MIN_DATA_ROWS}**")
            stage_placeholder.info(f"正在采集数据：{rows}/{MIN_DATA_ROWS}。数据达到要求后，系统将自动进入预测与喷淋控制阶段。")
        else:
            phase_placeholder.markdown("### 当前阶段: **智能控制中**")
            progress_placeholder.progress(1.0)
            stage_placeholder.success("数据采集已完成，系统已进入预测与喷淋控制阶段。")

        if df is not None and len(df) > 0:
            # 1. 更新数值指标
            latest = df.iloc[-1]

            # 2. 更新系统状态
            if rows >= MIN_DATA_ROWS:
                status_color = "🔴 喷淋开启" if cmd == "1" else "⚪ 系统待机"
                status_placeholder.markdown(f"### 当前状态: **{status_color}**")
            pm25_metric.metric("当前 PM2.5", f"{latest['PM2.5']} μg/m³")
            pm10_metric.metric("当前 PM10", f"{latest['PM10']} μg/m³")
            tsp_metric.metric("当前 TSP", f"{latest['TSP']} μg/m³")

            # 3. 更新动态折线图
            # 取最后 50 个数据点展示
            plot_df = df.tail(50)[['PM2.5', 'TSP']]
            chart_placeholder.line_chart(plot_df)

            # 4. 更新监控画面
            # collector.py 每次处理完帧后保存 latest_frame.jpg，app.py 直接读取它。
            frame_path = os.path.join(BASE_DIR, "latest_frame.jpg")
            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                video_placeholder.image(img, width="stretch")
        else:
            pm25_metric.metric("当前 PM2.5", "等待数据")
            pm10_metric.metric("当前 PM10", "等待数据")
            tsp_metric.metric("当前 TSP", "等待数据")

        time.sleep(0.5)  # 刷新频率


if get_script_run_ctx() is None:
    launch_system()
else:
    render_dashboard()
