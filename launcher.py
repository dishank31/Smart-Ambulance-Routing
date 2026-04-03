import subprocess
import time
import os
import sys

# Find the correct Python interpreter
# The .venv is at the parent directory (d:\ML Project\.venv)
project_root = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(os.path.dirname(project_root), ".venv", "Scripts", "python.exe")
if not os.path.exists(venv_python):
    venv_python = sys.executable  # fallback to current Python
print(f"Using Python: {venv_python}")

def run_backend():
    print("🚀 Starting FastAPI Backend...")
    return subprocess.Popen([venv_python, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"], cwd=project_root)

def run_frontend():
    print("🎨 Starting Streamlit Frontend...")
    return subprocess.Popen([venv_python, "-m", "streamlit", "run", "frontend/app_streamlit.py", "--server.port", "8501"], cwd=project_root)

if __name__ == "__main__":
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    backend_proc = None
    frontend_proc = None
    
    try:
        backend_proc = run_backend()
        # Give backend a moment to initialize
        time.sleep(5)
        
        frontend_proc = run_frontend()
        
        print("\n" + "="*50)
        print("✅ SYSTEM ONLINE")
        print("Backend: http://localhost:8000")
        print("Frontend: http://localhost:8501")
        print("="*50)
        print("\nPress Ctrl+C to stop both servers.")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_proc.poll() is not None:
                print("❌ Backend crashed. Exiting...")
                break
            if frontend_proc.poll() is not None:
                print("❌ Frontend crashed. Exiting...")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    finally:
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        print("Done.")
