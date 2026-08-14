"""Start the local PaperWeave development stack without rebuilding dependencies.

Run from the repository root:
    python run.py
"""

from __future__ import annotations

import argparse
import http.client
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
RUNTIME_IMAGE = "agentic-rag-project-main2-runtime:latest"
API_HEALTH_URL = "http://127.0.0.1:8059/health/live"
COMPOSE_COMMAND = ["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.dev.yml"]


def configure_console_encoding() -> None:
    """Keep Chinese startup diagnostics readable in a native Windows console."""
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            pass
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_command(command: Sequence[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    if not capture_output:
        print(f"> {' '.join(command)}")
    return subprocess.run(
        list(command),
        cwd=ROOT,
        check=False,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def http_is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 400
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        return False


def wait_for_http(url: str, timeout_seconds: int, service_name: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if http_is_ready(url):
            print(f"[ok] {service_name}: {url}")
            return
        time.sleep(2)
    raise RuntimeError(f"{service_name} did not become ready within {timeout_seconds}s: {url}")


def docker_is_ready() -> bool:
    return run_command(["docker", "info"], capture_output=True).returncode == 0


def docker_desktop_executable() -> Path | None:
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidate = program_files / "Docker" / "Docker" / "Docker Desktop.exe"
    return candidate if candidate.is_file() else None


def ensure_docker_ready(*, start_if_needed: bool) -> None:
    if docker_is_ready():
        return
    if not start_if_needed:
        raise RuntimeError("Docker Desktop 未就绪。请启动 Docker Desktop 后重试。")

    # Docker Desktop may already be launching after Windows login, so wait briefly first.
    startup_deadline = time.monotonic() + 8
    while time.monotonic() < startup_deadline:
        if docker_is_ready():
            return
        time.sleep(2)

    if os.name != "nt":
        raise RuntimeError("Docker daemon 未就绪。请启动 Docker 后重试。")

    executable = docker_desktop_executable()
    if executable is None:
        raise RuntimeError("未找到 Docker Desktop。请安装并启动 Docker Desktop 后重试。")

    print("[info] 正在启动 Docker Desktop，等待 Linux daemon 就绪…")
    subprocess.Popen(
        [str(executable)],
        cwd=executable.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )
    startup_deadline = time.monotonic() + 90
    while time.monotonic() < startup_deadline:
        if docker_is_ready():
            print("[ok] Docker Desktop 已就绪。")
            return
        time.sleep(3)
    raise RuntimeError("Docker Desktop 启动超时。请确认它已切换到 Linux containers 后重试。")


def check_prerequisites() -> None:
    require((ROOT / ".env").is_file(), "缺少 .env。请先从 .env.example 创建并填写配置。")
    require(WEB_DIR.is_dir(), f"找不到前端目录：{WEB_DIR}")
    require((WEB_DIR / "node_modules").is_dir(), "缺少 web/node_modules。本脚本不会下载依赖，请先完成一次前端依赖安装。")
    require(shutil.which("docker") is not None, "未找到 docker 命令。请启动 Docker Desktop 并确认命令行可用。")
    require(shutil.which("npm.cmd" if os.name == "nt" else "npm") is not None, "未找到 npm。请安装 Node.js 后重试。")

    image_check = run_command(["docker", "image", "inspect", RUNTIME_IMAGE], capture_output=True)
    require(
        image_check.returncode == 0,
        f"缺少运行时镜像 {RUNTIME_IMAGE}。本脚本不会构建镜像，请按 README 的首次部署流程构建一次。",
    )


def start_backend() -> None:
    # API and worker must be created separately to avoid the historical port-binding issue.
    for services in (("redis", "rabbitmq", "postgres"), ("api",), ("ingestion-worker",)):
        result = run_command([*COMPOSE_COMMAND, "up", "-d", "--no-build", *services])
        require(result.returncode == 0, f"启动服务失败：{', '.join(services)}")


def start_web(port: int) -> str:
    url = f"http://127.0.0.1:{port}/"
    if http_is_ready(url):
        print(f"[ok] 前端已在运行：{url}")
        return url

    log_dir = ROOT / ".tmp"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "web-dev.log"
    npm_command = "npm.cmd" if os.name == "nt" else "npm"
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.Popen(
            [npm_command, "run", "dev", "--", "--host", "0.0.0.0", "--port", str(port)],
            cwd=WEB_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            creationflags=creationflags,
        )

    wait_for_http(url, 30, "前端")
    return url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键启动 PaperWeave 本地开发环境，不下载或重建依赖。")
    parser.add_argument("--check", action="store_true", help="仅检查 Docker、镜像、.env 与前端依赖，不启动服务。")
    parser.add_argument("--no-browser", action="store_true", help="启动完成后不自动打开浏览器。")
    parser.add_argument("--port", type=int, default=5174, help="Vite 前端端口，默认 5174。")
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    try:
        require(shutil.which("docker") is not None, "未找到 docker 命令。请安装 Docker Desktop 后重试。")
        ensure_docker_ready(start_if_needed=not args.check)
        check_prerequisites()
        print("[ok] 启动前检查通过：不会下载依赖或重建镜像。")
        if args.check:
            return 0

        start_backend()
        wait_for_http(API_HEALTH_URL, 90, "API")
        web_url = start_web(args.port)
        print("\nPaperWeave 已启动")
        print(f"- 前端：{web_url}")
        print(f"- API：{API_HEALTH_URL}")
        print(f"- 前端日志：{ROOT / '.tmp' / 'web-dev.log'}")
        if not args.no_browser:
            webbrowser.open(web_url)
        return 0
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
