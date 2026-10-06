"""Explicitly launched deployment example; importing it does not rent a GPU."""

import os
from pathlib import Path

import modal

GPU = os.environ.get("MAI_MODAL_GPU", "RTX-PRO-6000")
if GPU not in ("RTX-PRO-6000", "B200", "B300"):
    raise ValueError("MAI_MODAL_GPU must be RTX-PRO-6000, B200 or B300.")
BASE_IMAGE = "nvcr.io/nvidia/pytorch:26.02-py3"
COMFYUI_REF = "fb2315f11db0ebfaafa9099a5df5227dc6bb42bc"
REPO = Path(__file__).resolve().parents[2]
image = (
    modal.Image.from_registry(BASE_IMAGE)
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(f"git clone https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI && git -C /opt/ComfyUI checkout {COMFYUI_REF}")
    .add_local_dir(str(REPO), "/opt/ComfyUI/custom_nodes/mAI_Utils_V02", copy=True,
                   ignore=[".git", ".pytest_cache", "**/__pycache__", "*.pt", "*.mp4", ".agents", ".codex"])
    .run_commands("python /opt/ComfyUI/custom_nodes/mAI_Utils_V02/deployment/auto_seamless_loop/install_dependencies.py")
)
app = modal.App("mai-auto-seamless-loop", image=image)
# from_name is lazy; the volume is created only when the user runs/deploys this app.
volume = modal.Volume.from_name("mai-comfy-workspace", create_if_missing=True)


@app.function(gpu=GPU, volumes={"/workspace": volume}, timeout=3600, max_containers=1)
def benchmark():
    import subprocess
    return subprocess.check_output([
        "python", "/opt/ComfyUI/custom_nodes/mAI_Utils_V02/scripts/benchmark_seamless_loop.py",
        "--device", "cuda", "--runs", "3",
    ], text=True)


@app.function(gpu=GPU, volumes={"/workspace": volume}, timeout=3600, max_containers=1)
@modal.web_server(8188, startup_timeout=180, requires_proxy_auth=True)
def comfyui():
    import subprocess
    import sys
    sys.path.insert(0, "/opt/ComfyUI/custom_nodes/mAI_Utils_V02")
    from utils.seamless_loop import cuda_smoke_test
    import torch
    # Fail early on incompatible provider/image combinations.
    if not cuda_smoke_test(torch.device("cuda:0"))["passed"]:
        raise RuntimeError("CUDA smoke test failed.")
    subprocess.Popen([sys.executable, "custom_nodes/mAI_Utils_V02/deployment/auto_seamless_loop/start_comfyui.py"], cwd="/opt/ComfyUI")


@app.local_entrypoint()
def main():
    print(benchmark.remote())
