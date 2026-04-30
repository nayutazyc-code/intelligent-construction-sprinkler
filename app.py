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

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREFERRED_PYTHON = "/Users/nayuta/miniconda3/envs/dust_env/bin/python"
RUNTIME_PYTHON = PREFERRED_PYTHON if os.path.exists(PREFERRED_PYTHON) else sys.executable
DATA_FILE = os.path.join(OUTPUT_DIR, "dust_dataset.csv")
COMMAND_FILE = os.path.join(OUTPUT_DIR, "cannon_command.txt")
DRL_SCRIPT = os.path.join(BASE_DIR, "drl_controller.py")
DASHBOARD_SCRIPT = os.path.join(BASE_DIR, "dashboard.py")
MIN_DATA_ROWS = 800
APP_PORT = 8502
SERVER_START_TIMEOUT = 60
MANAGED_PROCESSES = []


def dashboard_url(port):
    return f"http://localhost:{port}"


def health_check_url(port):
    return f"http://127.0.0.1:{port}"


def get_data_length():
    if not os.path.exists(DATA_FILE):
        return 0
    try:
        return len(pd.read_csv(DATA_FILE))
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


def streamlit_args(port=APP_PORT):
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


def open_dashboard_when_ready(port=APP_PORT):
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


def start_foreground_dashboard(port=APP_PORT):
    opener = threading.Thread(target=open_dashboard_when_ready, args=(port,), daemon=True)
    opener.start()

    proc = subprocess.Popen(streamlit_args(port), cwd=BASE_DIR, env=child_env())
    MANAGED_PROCESSES.append(("网页服务", proc))
    return proc


def start_managed_process(name, args, **kwargs):
    proc = subprocess.Popen(args, start_new_session=True, **kwargs)
    MANAGED_PROCESSES.append((name, proc))
    return proc


def child_env():
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["MPLCONFIGDIR"] = os.path.join(OUTPUT_DIR, "matplotlib")
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


def register_shutdown_handlers():
    def handle_shutdown(signum, frame):
        print("\n收到停止信号，正在关闭后台进程...")
        cleanup_processes()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_shutdown)


def launch_system():
    register_shutdown_handlers()
    atexit.register(cleanup_processes)

    print("=" * 60)
    print("智慧工地 AI 抑尘系统启动器")
    print("=" * 60)
    print(f"运行解释器: {RUNTIME_PYTHON}")

    print("\n[1] 启动 DRL 控制系统...")
    drl_proc = start_managed_process("DRL 控制系统", [RUNTIME_PYTHON, DRL_SCRIPT], cwd=BASE_DIR, env=child_env())

    print("\n[2] 启动网页监控面板...")
    print(f"网页地址: {dashboard_url(APP_PORT)}")
    print(f"[3] 页面将实时显示数据采集进度，达到 {MIN_DATA_ROWS} 行后进入控制阶段。")
    print("如果网页打不开，请直接查看下面 Streamlit 输出的 Traceback/Error。")

    try:
        dashboard_proc = start_foreground_dashboard(APP_PORT)
        dashboard_proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止系统...")
    finally:
        cleanup_processes()


if __name__ == "__main__":
    launch_system()
