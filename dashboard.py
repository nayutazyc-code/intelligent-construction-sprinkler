import os
import time

import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(OUTPUT_DIR, "dust_dataset.csv")
COMMAND_FILE = os.path.join(OUTPUT_DIR, "cannon_command.txt")
MIN_DATA_ROWS = 800


def render_dashboard():
    st.set_page_config(page_title="智慧工地 AI 抑尘系统", layout="wide")

    st.title("🏗️ 智慧工地：基于 DRL + LSTM 的抑尘自主控制云平台")

    st.sidebar.header("系统状态")
    phase_placeholder = st.sidebar.empty()
    progress_placeholder = st.sidebar.empty()
    status_placeholder = st.sidebar.empty()
    pm25_metric = st.sidebar.empty()
    pm10_metric = st.sidebar.empty()
    tsp_metric = st.sidebar.empty()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 实时 AI 监控画面")
        video_placeholder = st.empty()

    with col2:
        st.subheader("📈 指数实时动态")
        chart_placeholder = st.empty()

    stage_placeholder = st.empty()

    def get_data():
        if os.path.exists(DATA_FILE):
            try:
                return pd.read_csv(DATA_FILE)
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
            latest = df.iloc[-1]

            if rows >= MIN_DATA_ROWS:
                status_color = "🔴 喷淋开启" if cmd == "1" else "⚪ 系统待机"
                status_placeholder.markdown(f"### 当前状态: **{status_color}**")
            pm25_metric.metric("当前 PM2.5", f"{latest['PM2.5']} μg/m³")
            pm10_metric.metric("当前 PM10", f"{latest['PM10']} μg/m³")
            tsp_metric.metric("当前 TSP", f"{latest['TSP']} μg/m³")

            plot_df = df.tail(50)[['PM2.5', 'TSP']]
            chart_placeholder.line_chart(plot_df)

            frame_path = os.path.join(OUTPUT_DIR, "latest_frame.jpg")
            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                video_placeholder.image(img, width="stretch")
        else:
            pm25_metric.metric("当前 PM2.5", "等待数据")
            pm10_metric.metric("当前 PM10", "等待数据")
            tsp_metric.metric("当前 TSP", "等待数据")

        time.sleep(0.5)


render_dashboard()
