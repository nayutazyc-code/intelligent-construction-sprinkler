import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

import pandas as pd

from config import (
    DEFAULT_APP_PORT,
    BASE_DIR,
    archive_existing_runtime_files,
    is_config_ready,
    load_config,
    runtime_paths,
)

# --- 配置 ---
PREFERRED_PYTHON = "/Users/nayuta/miniconda3/envs/dust_env/bin/python"
RUNTIME_PYTHON = PREFERRED_PYTHON if os.path.exists(PREFERRED_PYTHON) else sys.executable
DRL_SCRIPT = os.path.join(BASE_DIR, "drl_controller.py")
DASHBOARD_SCRIPT = os.path.join(BASE_DIR, "dashboard.py")
SERVER_START_TIMEOUT = 60
MANAGED_PROCESSES = []
RUNTIME_PREPARED = False


def dashboard_url(port):
    return f"http://localhost:{port}"


def health_check_url(port):
    return f"http://127.0.0.1:{port}"


def get_data_length():
    paths = runtime_paths()
    if not os.path.exists(paths["data_file"]):
        return 0
    try:
        return len(pd.read_csv(paths["data_file"]))
    except Exception:
        return 0


def wait_for_server(proc, port, timeout=SERVER_START_TIMEOUT):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if proc.poll() is not None:
            return False

        try:
            with urllib.request.urlopen(health_check_url(port), timeout=1):
                return True
        except Exception:
            time.sleep(0.5)

    return False


def open_dashboard_url(url):
    opened = webbrowser.open(url, new=2)

    if sys.platform == "darwin":
        for command in (["/usr/bin/open", url], ["open", url]):
            try:
                subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                opened = True
                break
            except Exception:
                pass

    return opened


def streamlit_args(port):
    return [
        RUNTIME_PYTHON,
        "-m",
        "streamlit",
        "run",
        DASHBOARD_SCRIPT,
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "false",
        "--server.showEmailPrompt",
        "false",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]


def open_dashboard_when_ready(port):
    url = dashboard_url(port)
    start_time = time.time()
    while time.time() - start_time < SERVER_START_TIMEOUT:
        try:
            with urllib.request.urlopen(health_check_url(port), timeout=1):
                open_dashboard_url(url)
                print(f"网页已打开: {url}")
                return
        except Exception:
            time.sleep(0.5)

    print(f"未能自动检测到网页服务，请检查控制台报错，或手动打开: {url}")


def start_foreground_dashboard(port):
    opener = threading.Thread(target=open_dashboard_when_ready, args=(port,), daemon=True)
    opener.start()

    proc = subprocess.Popen(streamlit_args(port), cwd=BASE_DIR, env=child_env(), start_new_session=True)
    MANAGED_PROCESSES.append(("网页服务", proc))
    return proc


def start_managed_process(name, args, **kwargs):
    proc = subprocess.Popen(args, start_new_session=True, **kwargs)
    MANAGED_PROCESSES.append((name, proc))
    return proc


def child_env():
    paths = runtime_paths()
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["MPLCONFIGDIR"] = paths["matplotlib_dir"]
    if RUNTIME_PREPARED:
        env["SMART_SITE_RUNTIME_PREPARED"] = "1"
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    return env


def stop_process_group(name, proc):
    if proc.poll() is not None:
        return

    print(f"正在停止 {name}...")
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def cleanup_processes():
    for name, proc in reversed(MANAGED_PROCESSES):
        stop_process_group(name, proc)
    MANAGED_PROCESSES.clear()


def stop_process(name, proc):
    stop_process_group(name, proc)
    MANAGED_PROCESSES[:] = [(item_name, item_proc) for item_name, item_proc in MANAGED_PROCESSES if item_proc != proc]


def run_initial_setup():
    print("\n检测到初始化配置缺失或无效，先打开初始化设置页面...")
    print(f"初始化页面地址: {dashboard_url(DEFAULT_APP_PORT)}")
    print("请在网页中保存设置。保存后系统会自动进入启动流程。")

    dashboard_proc = start_foreground_dashboard(DEFAULT_APP_PORT)
    try:
        while dashboard_proc.poll() is None:
            if is_config_ready():
                print("\n初始化配置已保存，正在切换到完整系统...")
                stop_process("初始化设置页面", dashboard_proc)
                return True
            time.sleep(1)
    except KeyboardInterrupt:
        raise

    return is_config_ready()


def register_shutdown_handlers():
    def handle_shutdown(signum, frame):
        print("\n收到停止信号，正在关闭后台进程...")
        cleanup_processes()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_shutdown)


def launch_system():
    global RUNTIME_PREPARED

    register_shutdown_handlers()
    atexit.register(cleanup_processes)

    print("=" * 60)
    print("智慧工地 AI 抑尘系统启动器")
    print("=" * 60)
    print(f"运行解释器: {RUNTIME_PYTHON}")

    if not is_config_ready() and not run_initial_setup():
        print("初始化配置尚未完成，系统未启动。")
        return

    config = load_config()
    app_port = config["app_port"]
    min_data_rows = config["min_data_rows"]
    archive_dir, moved_files = archive_existing_runtime_files(config)
    RUNTIME_PREPARED = True

    if moved_files:
        print(f"\n已归档历史运行文件: {archive_dir}")
        print(f"本次启动将从空数据状态开始，共归档 {len(moved_files)} 个文件。")

    print("\n[1] 启动 DRL 控制系统...")
    drl_proc = start_managed_process("DRL 控制系统", [RUNTIME_PYTHON, DRL_SCRIPT], cwd=BASE_DIR, env=child_env())

    print("\n[2] 启动网页监控面板...")
    print(f"网页地址: {dashboard_url(app_port)}")
    print(f"[3] 页面将实时显示数据采集进度，达到 {min_data_rows} 行后进入控制阶段。")
    print("如果网页打不开，请直接查看下面 Streamlit 输出的 Traceback/Error。")

    try:
        dashboard_proc = start_foreground_dashboard(app_port)
        dashboard_proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止系统...")
    finally:
        cleanup_processes()


if __name__ == "__main__":
    launch_system()
