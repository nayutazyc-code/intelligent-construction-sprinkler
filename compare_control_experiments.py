import argparse
import os


PM25_SAFE_THRESHOLD = 75.0
TSP_SAFE_THRESHOLD = 200.0
SPRAY_WATER_RATE = {
    0: 0.0,
    1: 1.0,
    2: 1.8,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare control logs from 2 or 3 spray strategies.")
    parser.add_argument("--log", action="append", default=[],
                        help="Control log in the form 'Strategy Label=/path/to/control_log.csv'. "
                             "Pass this option 2 or 3 times.")
    parser.add_argument("--dqn-log", default=None, help="Backward-compatible path to dqn_control_log.csv")
    parser.add_argument("--baseline-log", default=None,
                        help="Backward-compatible path to threshold_baseline_control_log.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison outputs")
    return parser.parse_args()


def parse_log_specs(args):
    if args.log:
        specs = []
        for item in args.log:
            if "=" not in item:
                raise ValueError(f"--log 参数必须使用 '策略名称=日志路径' 格式: {item}")
            label, path = item.split("=", 1)
            label = label.strip()
            path = path.strip()
            if not label or not path:
                raise ValueError(f"--log 参数必须包含非空策略名称和路径: {item}")
            specs.append((label, path))
    elif args.dqn_log and args.baseline_log:
        specs = [
            ("DQN", args.dqn_log),
            ("Threshold Baseline", args.baseline_log),
        ]
    else:
        raise ValueError("请提供 2 到 3 个 --log，或同时提供 --dqn-log 与 --baseline-log。")

    if not 2 <= len(specs) <= 3:
        raise ValueError(f"对比脚本需要 2 到 3 个控制日志，当前收到 {len(specs)} 个。")
    return specs


def read_log(path):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path)
    required_columns = ["PM2.5", "TSP", "final_action"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必要列: {', '.join(missing)}")
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df["TSP"] = pd.to_numeric(df["TSP"], errors="coerce")
    df["final_action"] = pd.to_numeric(df["final_action"], errors="coerce").fillna(0).astype(int)
    if "step" not in df.columns:
        df["step"] = np.arange(1, len(df) + 1)
    return df.dropna(subset=["PM2.5", "TSP"]).reset_index(drop=True)


def switch_count(df):
    import pandas as pd

    if "switched" in df.columns:
        return int(pd.to_numeric(df["switched"], errors="coerce").fillna(0).sum())
    return int(df["final_action"].diff().abs().fillna(0).sum())


def average_switch_interval_seconds(df):
    import numpy as np
    import pandas as pd

    if "switched" in df.columns:
        switched_df = df[pd.to_numeric(df["switched"], errors="coerce").fillna(0) == 1]
    else:
        switched_df = df[df["final_action"].diff().abs().fillna(0) == 1]

    if len(switched_df) < 2:
        return np.nan

    if "timestamp" in switched_df.columns:
        timestamps = pd.to_numeric(switched_df["timestamp"], errors="coerce").dropna()
        if len(timestamps) >= 2:
            return float(timestamps.diff().dropna().mean())

    return float(pd.to_numeric(switched_df["step"], errors="coerce").diff().dropna().mean())


def estimate_response_delay(df):
    import numpy as np

    polluted = ((df["PM2.5"] > PM25_SAFE_THRESHOLD) | (df["TSP"] > TSP_SAFE_THRESHOLD)).to_numpy()
    spraying = (df["final_action"] > 0).to_numpy()
    delays = []
    in_event = False
    event_start = None

    for index, is_polluted in enumerate(polluted):
        if is_polluted and not in_event:
            in_event = True
            event_start = index
        elif not is_polluted:
            in_event = False
            event_start = None

        if in_event and event_start is not None and spraying[index]:
            delays.append(index - event_start)
            event_start = None

    return float(np.mean(delays)) if delays else np.nan


def summarize(strategy, df):
    import numpy as np

    polluted = (df["PM2.5"] > PM25_SAFE_THRESHOLD) | (df["TSP"] > TSP_SAFE_THRESHOLD)
    pm25_excess = np.maximum(df["PM2.5"] - PM25_SAFE_THRESHOLD, 0)
    tsp_excess = np.maximum(df["TSP"] - TSP_SAFE_THRESHOLD, 0)
    spray_levels = df["final_action"].clip(lower=0, upper=2)
    estimated_water_usage = float(spray_levels.map(SPRAY_WATER_RATE).sum())
    continuous_high_water = len(df) * SPRAY_WATER_RATE[2]
    return {
        "strategy": strategy,
        "steps": int(len(df)),
        "avg_pm25": float(df["PM2.5"].mean()),
        "avg_tsp": float(df["TSP"].mean()),
        "high_pollution_ratio": float(polluted.mean()),
        "exceed_duration_steps": int(polluted.sum()),
        "pm25_exceed_area": float(pm25_excess.sum()),
        "tsp_exceed_area": float(tsp_excess.sum()),
        "total_exceed_area": float(pm25_excess.sum() + tsp_excess.sum()),
        "spray_duty_ratio": float((spray_levels > 0).mean()),
        "avg_spray_level": float(spray_levels.mean()),
        "total_spray_steps": int((spray_levels > 0).sum()),
        "estimated_water_usage": estimated_water_usage,
        "water_saving_ratio_vs_continuous_high": (
            float(1 - estimated_water_usage / continuous_high_water)
            if continuous_high_water > 0 else np.nan
        ),
        "response_delay_steps": estimate_response_delay(df),
        "switch_count": switch_count(df),
        "avg_switch_interval_seconds": average_switch_interval_seconds(df),
    }


def add_unit_spray_benefit(metrics):
    import numpy as np

    worst_exceed_area = float(metrics["total_exceed_area"].max())
    metrics["exceed_area_reduction_vs_worst"] = worst_exceed_area - metrics["total_exceed_area"]
    metrics["unit_spray_benefit"] = np.where(
        metrics["estimated_water_usage"] > 0,
        metrics["exceed_area_reduction_vs_worst"] / metrics["estimated_water_usage"],
        np.nan,
    )
    return metrics


def save_comparison_plot(logs, output_file):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("未安装 matplotlib，已跳过对比图生成。")
        return False

    plt.figure(figsize=(15, 10))
    colors = ["#e74c3c", "#2c7fb8", "#16a085"]

    plt.subplot(3, 1, 1)
    for index, (label, df) in enumerate(logs):
        plt.plot(df["step"], df["PM2.5"], label=f"{label} PM2.5",
                 color=colors[index % len(colors)], alpha=0.85)
    plt.axhline(PM25_SAFE_THRESHOLD, color="green", linestyle="--", label="PM2.5 Limit")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    for index, (label, df) in enumerate(logs):
        plt.plot(df["step"], df["TSP"], label=f"{label} TSP",
                 color=colors[index % len(colors)], alpha=0.85)
    plt.axhline(TSP_SAFE_THRESHOLD, color="orange", linestyle="--", label="TSP Limit")
    plt.ylabel("TSP")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    for index, (label, df) in enumerate(logs):
        plt.step(df["step"], df["final_action"], where="post", label=f"{label} Spray",
                 color=colors[index % len(colors)])
    plt.yticks([0, 1, 2], ["Off", "Low", "High"])
    plt.xlabel("Control Step")
    plt.ylabel("Spray")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    return True


def main():
    args = parse_args()
    import pandas as pd

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(args.output_dir, "matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    log_specs = parse_log_specs(args)
    logs = [(label, read_log(path)) for label, path in log_specs]

    metrics = pd.DataFrame([summarize(label, df) for label, df in logs])
    metrics = add_unit_spray_benefit(metrics)
    metrics_file = os.path.join(args.output_dir, "control_comparison_metrics.csv")
    metrics.to_csv(metrics_file, index=False)

    plot_file = os.path.join(args.output_dir, "control_comparison.png")
    plot_created = save_comparison_plot(logs, plot_file)

    print(f"对比指标已生成: {metrics_file}")
    if plot_created:
        print(f"对比图已生成: {plot_file}")


if __name__ == "__main__":
    main()
