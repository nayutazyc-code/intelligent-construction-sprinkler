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
├── collector.py           # 视频采集、YOLO 检测和环境数据模拟
├── drl_controller.py      # DQN 喷淋控制逻辑
├── 预测1.py               # Attention-LSTM 预测模型训练脚本
├── best.pt                # YOLO 检测模型权重
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

2. 修改 `collector.py` 中的视频路径：

```python
VIDEO_PATH = "/Users/nayuta/Desktop/data3.mp4"
```

将其改为本机实际的视频文件路径。

3. 启动系统：

```bash
streamlit run app.py
```

系统默认会在本地启动 Streamlit 页面，并实时展示粉尘检测、指标变化和喷淋状态。

## 运行过程说明

首次运行时，系统会先启动数据采集与环境模拟流程，等待数据量达到训练要求后，再训练或加载预测模型，并进入喷淋控制流程。

运行过程中可能会生成以下文件：

- `dust_dataset.csv`
- `cannon_command.txt`
- `latest_frame.jpg`
- `dust_attention_lstm_model.keras`
- `optimized_prediction.png`
- `attention_heatmap.png`
- `analysis_result.png`
- `drl_multi_metrics_evaluation.png`

这些文件属于运行数据、模型输出或实验结果，可根据需要自行保存。

## 注意事项

- `best.pt` 是目标检测权重文件，运行前需要确保文件存在。
- `collector.py` 中的视频路径需要根据本机环境修改。
- 如果 OpenCV 无法打开视频，请优先检查视频路径是否正确。
- TensorFlow 和 Ultralytics 安装耗时较长，建议在虚拟环境中配置项目依赖。
