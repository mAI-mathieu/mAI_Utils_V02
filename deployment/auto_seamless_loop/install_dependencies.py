"""Cloud image BUILD helper. Preserve the image's tested torch packages."""

from pathlib import Path
import subprocess
import sys

import torch
import torchvision

root = Path("/opt/ComfyUI")
constraints = Path("/tmp/mai-torch-constraints.txt")
constraints.write_text(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n", encoding="utf-8")
filtered = Path("/tmp/mai-comfy-requirements.txt")
lines = root.joinpath("requirements.txt").read_text().splitlines()
filtered.write_text("\n".join(line for line in lines if line.strip() not in ("torch", "torchvision", "torchaudio")) + "\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-c", str(constraints), "-r", str(filtered),
                "-r", str(root / "custom_nodes/mAI_Utils_V02/requirements.txt")], check=True)
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
root.joinpath("image-requirements.lock.txt").write_text(freeze, encoding="utf-8")
# Fail at build time if the dependency resolver replaced torch.
subprocess.run([sys.executable, "-c", f"import torch; assert torch.__version__ == {torch.__version__!r}"], check=True)
