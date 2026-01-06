import uvicorn
from pathlib import Path
import subprocess
import sys, os
import webbrowser
import threading
import time
from urllib import request

root = Path(__file__).resolve().parent

def open_when_ready(url="http://localhost:8000", timeout=15):
    end = time.time() + timeout
    
    while time.time() < end:
        try:
            request.urlopen(url, timeout=0.2)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)

    print("Timeout: Server could not be open!")


def run_webapp():
    uvicorn.run(
        app="app.main:app",
        host="127.0.0.1",
        port=8000,
        reload_dirs=[(root / 'app').absolute().as_posix(), (root / 'src').absolute().as_posix()],
        reload=True,
    )

def run_tranning():
    args = sys.argv[1:]

    env = os.environ.copy()

    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, str(root / "scripts" / "model_setup.py"), *args],
                   env=env,
                   check=True)

if __name__ == '__main__':
    run_tranning()
    threading.Thread(target=open_when_ready, daemon=True).start()
    run_webapp()