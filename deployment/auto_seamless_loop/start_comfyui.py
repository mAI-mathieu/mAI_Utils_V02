"""Container entrypoint: initialize the mounted workspace, then exec ComfyUI."""

import os
from pathlib import Path
import sys

for folder in ("input", "output", "user"):
    Path("/workspace", folder).mkdir(parents=True, exist_ok=True)
os.chdir("/opt/ComfyUI")
os.execv(sys.executable, [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188",
                         "--input-directory", "/workspace/input", "--output-directory", "/workspace/output",
                         "--user-directory", "/workspace/user", "--disable-auto-launch", *sys.argv[1:]])
