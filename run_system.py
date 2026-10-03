import os
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def find_python() -> str:
    local_venv = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    parent_venv = PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe"
    if local_venv.exists():
        return str(local_venv)
    if parent_venv.exists():
        return str(parent_venv)
    return sys.executable


def find_open_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found near {preferred}")


def run_backend(python_exe: str, backend_port: int) -> subprocess.Popen:
    print(f"Starting FastAPI backend on port {backend_port}...")
    return subprocess.Popen(
        [python_exe, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
        cwd=PROJECT_ROOT,
    )


def run_frontend(python_exe: str, backend_port: int, frontend_port: int) -> subprocess.Popen:
    print(f"Starting Streamlit frontend on port {frontend_port}...")
    env = os.environ.copy()
    env["SMART_AMBULANCE_API"] = f"http://127.0.0.1:{backend_port}"
    return subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", "frontend/app_streamlit.py", "--server.port", str(frontend_port)],
        cwd=PROJECT_ROOT,
        env=env,
    )


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    python_exe = find_python()
    backend_port = find_open_port(8000)
    frontend_port = find_open_port(8501)

    print(f"Using Python: {python_exe}")

    backend_proc = None
    frontend_proc = None

    try:
        backend_proc = run_backend(python_exe, backend_port)
        time.sleep(3)
        if backend_proc.poll() is not None:
            raise RuntimeError("Backend exited during startup.")

        frontend_proc = run_frontend(python_exe, backend_port, frontend_port)
        time.sleep(3)
        if frontend_proc.poll() is not None:
            raise RuntimeError("Frontend exited during startup.")

        print("\n" + "=" * 50)
        print("SYSTEM ONLINE")
        print(f"Backend:  http://127.0.0.1:{backend_port}")
        print(f"Docs:     http://127.0.0.1:{backend_port}/docs")
        print(f"Frontend: http://127.0.0.1:{frontend_port}")
        print("=" * 50)
        print("\nPress Ctrl+C to stop both servers.")

        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                raise RuntimeError("Backend crashed.")
            if frontend_proc.poll() is not None:
                raise RuntimeError("Frontend crashed.")
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        if backend_proc and backend_proc.poll() is None:
            backend_proc.terminate()
        if frontend_proc and frontend_proc.poll() is None:
            frontend_proc.terminate()
        print("Done.")
