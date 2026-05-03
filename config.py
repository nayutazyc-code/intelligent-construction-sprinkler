import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
DEFAULT_VIDEO_PATH = "/Users/nayuta/Desktop/data3.mp4"
DEFAULT_MIN_DATA_ROWS = 800
DEFAULT_APP_PORT = 8502

DEFAULT_CONFIG = {
    "video_path": DEFAULT_VIDEO_PATH,
    "model_path": DEFAULT_MODEL_PATH,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "min_data_rows": DEFAULT_MIN_DATA_ROWS,
    "app_port": DEFAULT_APP_PORT,
}


def resolve_path(path):
    if not path:
        return ""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(BASE_DIR, expanded))


def normalize_config(config):
    normalized = DEFAULT_CONFIG.copy()
    if isinstance(config, dict):
        normalized.update(config)

    normalized["video_path"] = resolve_path(normalized.get("video_path"))
    normalized["model_path"] = resolve_path(normalized.get("model_path"))
    normalized["output_dir"] = resolve_path(normalized.get("output_dir"))

    try:
        normalized["min_data_rows"] = int(normalized.get("min_data_rows", DEFAULT_MIN_DATA_ROWS))
    except (TypeError, ValueError):
        normalized["min_data_rows"] = DEFAULT_MIN_DATA_ROWS

    try:
        normalized["app_port"] = int(normalized.get("app_port", DEFAULT_APP_PORT))
    except (TypeError, ValueError):
        normalized["app_port"] = DEFAULT_APP_PORT

    return normalized


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return normalize_config(json.load(f))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CONFIG.copy()


def config_exists():
    return os.path.exists(CONFIG_FILE)


def validate_config(config, require_existing_output=False):
    config = normalize_config(config)
    errors = []

    if not config["video_path"]:
        errors.append("视频文件路径不能为空。")
    elif not os.path.isfile(config["video_path"]):
        errors.append("视频文件不存在。")

    if not config["model_path"]:
        errors.append("YOLO 模型路径不能为空。")
    elif not os.path.isfile(config["model_path"]):
        errors.append("YOLO 模型文件不存在。")

    if not config["output_dir"]:
        errors.append("输出目录不能为空。")
    elif require_existing_output and not os.path.isdir(config["output_dir"]):
        errors.append("输出目录不存在。")

    if config["min_data_rows"] <= 0:
        errors.append("采集行数必须为正整数。")

    if not 1024 <= config["app_port"] <= 65535:
        errors.append("网页端口必须在 1024-65535 范围内。")

    return errors


def is_config_ready():
    return config_exists() and not validate_config(load_config())


def save_config(config):
    config = normalize_config(config)
    errors = validate_config(config)
    if errors:
        return False, errors

    os.makedirs(config["output_dir"], exist_ok=True)
    temp_path = CONFIG_FILE + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, CONFIG_FILE)
    return True, []


def runtime_paths(config=None):
    config = normalize_config(config or load_config())
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    return {
        "output_dir": output_dir,
        "data_file": os.path.join(output_dir, "dust_dataset.csv"),
        "command_file": os.path.join(output_dir, "cannon_command.txt"),
        "status_file": os.path.join(output_dir, "system_status.json"),
        "latest_frame_file": os.path.join(output_dir, "latest_frame.jpg"),
        "model_file": os.path.join(output_dir, "dust_attention_lstm_model.keras"),
        "prediction_plot_file": os.path.join(output_dir, "optimized_prediction.png"),
        "attention_heatmap_file": os.path.join(output_dir, "attention_heatmap.png"),
        "analysis_plot_file": os.path.join(output_dir, "analysis_result.png"),
        "evaluation_plot_file": os.path.join(output_dir, "drl_multi_metrics_evaluation.png"),
        "matplotlib_dir": os.path.join(output_dir, "matplotlib"),
    }


def runtime_artifact_paths(config=None):
    paths = runtime_paths(config)
    return [
        paths["data_file"],
        paths["command_file"],
        paths["status_file"],
        paths["latest_frame_file"],
        paths["model_file"],
        paths["prediction_plot_file"],
        paths["attention_heatmap_file"],
        paths["analysis_plot_file"],
        paths["evaluation_plot_file"],
    ]


def archive_existing_runtime_files(config=None):
    paths = runtime_paths(config)
    existing_files = [path for path in runtime_artifact_paths(config) if os.path.isfile(path)]
    if not existing_files:
        return None, []

    archive_dir = os.path.join(paths["output_dir"], "history", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(archive_dir, exist_ok=True)

    moved_files = []
    for path in existing_files:
        target_path = os.path.join(archive_dir, os.path.basename(path))
        os.replace(path, target_path)
        moved_files.append(target_path)

    return archive_dir, moved_files
