import json
import os
import time

import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    DEFAULT_APP_PORT,
    DEFAULT_MIN_DATA_ROWS,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_PATH,
    config_exists,
    is_config_ready,
    load_config,
    runtime_paths,
    save_config,
    validate_config,
)


def render_settings_form(config, initial_setup=False):
    st.title("系统初始化设置" if initial_setup else "系统设置")
    st.caption("保存后会写入 config.json，之后运行系统时将自动读取这些设置。")

    with st.form("settings_form"):
        video_path = st.text_input("施工现场视频路径", value=config.get("video_path") or DEFAULT_VIDEO_PATH)
        model_path = st.text_input("YOLO 模型路径", value=config.get("model_path") or DEFAULT_MODEL_PATH)
        output_dir = st.text_input("输出目录", value=config.get("output_dir") or DEFAULT_OUTPUT_DIR)
        min_data_rows = st.number_input(
            "进入训练阶段所需采集行数",
            min_value=1,
            step=50,
            value=int(config.get("min_data_rows") or DEFAULT_MIN_DATA_ROWS),
        )
        app_port = st.number_input(
            "网页端口",
            min_value=1024,
            max_value=65535,
            step=1,
            value=int(config.get("app_port") or DEFAULT_APP_PORT),
        )

        submitted = st.form_submit_button("保存设置", type="primary")

    draft = {
        "video_path": video_path,
        "model_path": model_path,
        "output_dir": output_dir,
        "min_data_rows": int(min_data_rows),
        "app_port": int(app_port),
    }

    errors = validate_config(draft)
    if submitted:
        ok, save_errors = save_config(draft)
        if ok:
            st.success("初始化设置已保存。系统启动器会自动进入运行流程；端口修改将在下次启动时生效。")
            st.session_state["show_settings"] = False
            time.sleep(1)
            st.rerun()
        else:
            for error in save_errors:
                st.error(error)
    elif errors:
        for error in errors:
            st.warning(error)


def get_runtime_state():
    config = load_config()
    paths = runtime_paths(config)
    return config, paths


def render_dashboard():
    st.set_page_config(page_title="智慧工地 AI 抑尘系统", layout="wide")

    config = load_config()
    ready = is_config_ready()
    if "show_settings" not in st.session_state:
        st.session_state["show_settings"] = not ready

    if not ready or st.session_state["show_settings"]:
        render_settings_form(config, initial_setup=not config_exists() or not ready)
        return

    config, paths = get_runtime_state()
    output_dir = paths["output_dir"]
    data_file = paths["data_file"]
    command_file = paths["command_file"]
    status_file = paths["status_file"]
    min_data_rows = config["min_data_rows"]

    st.title("智慧工地：基于 DRL + LSTM 的抑尘自主控制云平台")

    st.sidebar.header("系统状态")
    if st.sidebar.button("初始化设置 / 修改设置"):
        st.session_state["show_settings"] = True
        st.rerun()

    phase_placeholder = st.sidebar.empty()
    progress_placeholder = st.sidebar.empty()
    status_placeholder = st.sidebar.empty()
    pm25_metric = st.sidebar.empty()
    pm10_metric = st.sidebar.empty()
    tsp_metric = st.sidebar.empty()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("实时 AI 监控画面")
        video_placeholder = st.empty()

    with col2:
        st.subheader("指数实时动态")
        chart_placeholder = st.empty()

    stage_placeholder = st.empty()

    def get_data():
        if os.path.exists(data_file):
            try:
                return pd.read_csv(data_file)
            except Exception:
                return None
        return None

    def get_command():
        if os.path.exists(command_file):
            with open(command_file, "r") as f:
                return f.read().strip()
        return "0"

    def get_status():
        if os.path.exists(status_file):
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    while True:
        df = get_data()
        cmd = get_command()
        status = get_status()
        rows = 0 if df is None else len(df)
        progress_value = min(rows / min_data_rows, 1.0)
        stage = status.get("stage")
        message = status.get("message")

        if stage == "training":
            phase_placeholder.markdown("### 当前阶段: **模型训练中**")
            progress_placeholder.progress(1.0)
            status_placeholder.markdown("### Attention-LSTM 模型训练中")
            stage_placeholder.warning(message or "数据采集完成，正在训练 Attention-LSTM 预测模型。")
        elif stage == "control":
            phase_placeholder.markdown("### 当前阶段: **DRL 喷淋控制中**")
            progress_placeholder.progress(1.0)
            stage_placeholder.success(message or "模型已就绪，系统已进入 DRL 喷淋控制阶段。")
        elif stage == "error":
            phase_placeholder.markdown("### 当前阶段: **系统异常**")
            progress_placeholder.progress(progress_value)
            status_placeholder.markdown("### 请检查后台输出")
            stage_placeholder.error(message or "系统运行异常，请检查控制台输出。")
        elif stage == "stopped":
            phase_placeholder.markdown("### 当前阶段: **系统已停止**")
            progress_placeholder.progress(progress_value)
            status_placeholder.markdown("### 系统停止运行")
            stage_placeholder.info(message or "系统已停止。")
        elif rows < min_data_rows:
            phase_placeholder.markdown("### 当前阶段: **数据采集中**")
            progress_placeholder.progress(progress_value)
            status_placeholder.markdown(f"### 正在采集数据: **{rows}/{min_data_rows}**")
            stage_placeholder.info(message or f"正在采集数据：{rows}/{min_data_rows}。数据达到要求后，系统将进入 Attention-LSTM 模型训练阶段。")
        else:
            phase_placeholder.markdown("### 当前阶段: **准备进入模型训练**")
            progress_placeholder.progress(1.0)
            stage_placeholder.info("数据采集已完成，正在等待后台进入 Attention-LSTM 模型训练阶段。")

        if df is not None and len(df) > 0:
            latest = df.iloc[-1]

            if stage == "control":
                status_color = "🟢 喷淋开启" if cmd == "1" else "⚪ 系统待机"
                status_placeholder.markdown(f"### 当前状态: **{status_color}**")
            pm25_metric.metric("当前 PM2.5", f"{latest['PM2.5']} μg/m³")
            pm10_metric.metric("当前 PM10", f"{latest['PM10']} μg/m³")
            tsp_metric.metric("当前 TSP", f"{latest['TSP']} μg/m³")

            plot_df = df.tail(50)[['PM2.5', 'TSP']]
            chart_placeholder.line_chart(plot_df)

            frame_path = os.path.join(output_dir, "latest_frame.jpg")
            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                video_placeholder.image(img, width="stretch")
        else:
            pm25_metric.metric("当前 PM2.5", "等待数据")
            pm10_metric.metric("当前 PM10", "等待数据")
            tsp_metric.metric("当前 TSP", "等待数据")

        time.sleep(0.5)


render_dashboard()
