# 智能工地喷淋系统

本项目是一个面向智慧工地场景的 AI 抑尘喷淋控制系统。系统通过视频画面识别扬尘相关目标，模拟 PM2.5、PM10、TSP 等环境数据，并结合 LSTM 预测模型与 DQN 强化学习策略，自动判断是否开启喷淋设备。

## 功能简介

- 基于 YOLO 权重文件 `best.pt` 进行施工现场视频目标检测
- 模拟生成 PM2.5、PM10、TSP 等粉尘浓度数据
- 使用 Attention-LSTM 模型预测空气质量变化趋势
- 使用 DQN 强化学习算法进行喷淋开关决策
- 通过 Streamlit 页面实时展示监控画面、粉尘指标和喷淋状态

## 项目结构

```text
.
├── app.py                 # Streamlit 可视化界面与系统启动入口
├── dashboard.py           # Streamlit 实时监控页面
├── config.py              # 初始化设置读取、保存与校验
├── config.json            # 本机运行配置
├── collector.py           # 视频采集、YOLO 检测和环境数据模拟
├── drl_controller.py      # DQN 喷淋控制逻辑
├── 预测1.py               # Attention-LSTM 预测模型训练脚本
├── best.pt                # YOLO 检测模型权重
├── outputs/               # 运行时生成的数据、模型和图表
└── plots/                 # 相关图表或实验结果目录
```

## 环境依赖

建议使用 Python 3.9 及以上版本。主要依赖包括：

- streamlit
- pandas
- numpy
- pillow
- opencv-python
- ultralytics
- tensorflow
- scikit-learn
- matplotlib

可以根据本地环境安装：

```bash
pip install streamlit pandas numpy pillow opencv-python ultralytics tensorflow scikit-learn matplotlib
```

## 运行方式

1. 确认当前目录下存在 YOLO 权重文件：

```text
best.pt
```

2. 启动系统：

```bash
python app.py
```

首次运行会先打开 Streamlit 初始化设置页面，请填写：

- 施工现场视频路径
- YOLO 模型路径，默认使用当前项目下的 `best.pt`
- 输出目录，默认使用 `outputs/`
- 进入训练阶段所需采集行数，默认 `800`
- 网页端口，默认 `8502`

保存设置后会写入 `config.json`，系统启动器会自动进入采集、训练、控制和监控流程。后续运行会直接读取配置，不需要再修改源码。

## 运行过程说明

首次运行时，系统会先启动数据采集与环境模拟流程，等待数据量达到训练要求后，再训练或加载预测模型，并进入喷淋控制流程。

运行过程中生成的文件会统一保存到 `outputs/` 目录：

- `outputs/dust_dataset.csv`
- `outputs/cannon_command.txt`
- `outputs/latest_frame.jpg`
- `outputs/dust_attention_lstm_model.keras`
- `outputs/optimized_prediction.png`
- `outputs/attention_heatmap.png`
- `outputs/analysis_result.png`
- `outputs/drl_multi_metrics_evaluation.png`

这些文件属于运行数据、模型输出或实验结果，可根据需要自行保存。

## 注意事项

- `best.pt` 是目标检测权重文件，运行前需要确保文件存在。
- 视频路径、模型路径、输出目录、采集行数和网页端口都可以在首次初始化页面中设置。
- 如果需要重新配置，可在监控页面侧边栏点击“初始化设置 / 修改设置”。
- 如果 OpenCV 无法打开视频，请优先检查视频路径是否正确。
- TensorFlow 和 Ultralytics 安装耗时较长，建议在虚拟环境中配置项目依赖。
