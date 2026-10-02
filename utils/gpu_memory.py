"""Release unused CUDA allocations and request model offloading via ComfyUI.

Live input/output tensors remain owned by the workflow. No CUDA queries occur
at import time or for CPU operations; ComfyUI hooks are resolved at execution.
"""

from contextlib import contextmanager
import gc
import sys
import traceback

import torch


GIB = 1024 ** 3
NVENC_HEADROOM_BYTES = 2 * GIB


def release_gpu_cache(device):
    device = torch.device(device)
    if device.type != "cuda":
        return
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def prepare_gpu_memory(device, required_bytes=0, force_offload=False):
    device = torch.device(device)
    if device.type != "cuda":
        return
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    release_gpu_cache(device)
    management = sys.modules.get("comfy.model_management")
    free_memory = getattr(management, "free_memory", None)
    if free_memory is None:
        return
    free, _ = torch.cuda.mem_get_info(device)
    if force_offload or free < required_bytes:
        # ComfyUI counts reclaimable PyTorch cache as free memory. FFmpeg's
        # separate CUDA context needs driver-visible free bytes instead.
        cached = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
        free_memory(1e30 if force_offload else required_bytes + cached, device)
        release_gpu_cache(device)


@contextmanager
def gpu_memory_scope(device, required_bytes=0, force_offload=False):
    device = torch.device(device)
    if device.type != "cuda":
        yield
        return
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        prepare_gpu_memory(device, required_bytes, force_offload)
        yield
    except BaseException as exc:
        # Failed workers may retain GPU scratch tensors through traceback
        # locals. Keep stack locations while releasing those stopped frames.
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        release_gpu_cache(device)


def video_memory_requirements(frame_device, gpu_device, conversion_bytes):
    """Return CUDA device budgets; don't initialize CUDA for CPU-only tests.

    FFmpeg NVENC indices address visible CUDA devices. Automatic NVENC selection
    starts at the first visible device, which can differ from ComfyUI's device.
    """
    frame_device = torch.device(frame_device)
    management = sys.modules.get("comfy.model_management")
    preferred = getattr(management, "get_torch_device", None)
    runtime_device = torch.device(preferred()) if preferred else frame_device
    encoder_device = torch.device("cpu")
    if frame_device.type == "cuda" or runtime_device.type == "cuda":
        index = 0 if gpu_device == -1 else gpu_device
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            encoder_device = torch.device("cuda", index)
    budgets = {}
    for device, budget in ((frame_device, conversion_bytes), (encoder_device, NVENC_HEADROOM_BYTES)):
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            budgets[device] = budgets.get(device, 0) + budget
    return budgets
