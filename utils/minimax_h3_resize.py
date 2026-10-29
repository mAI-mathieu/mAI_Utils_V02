"""GPU-first Minimax resizing with a consistent tensor-only CPU fallback."""

import sys
import warnings

import torch

from .resize_kernels import resize_image_batch


def resolve_resize_device(image_device):
    # ComfyUI loads model management at startup. Avoid importing it here so
    # standalone use and CPU-only imports do not initialize CUDA.
    management = sys.modules.get("comfy.model_management")
    if management is not None:
        preferred = torch.device(management.get_torch_device())
        if preferred.type != "cuda":
            return torch.device("cpu")
    else:
        preferred = image_device if image_device.type == "cuda" else torch.device("cuda")
    return preferred if torch.cuda.is_available() else torch.device("cpu")


def _is_cuda_failure(error):
    message = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or any(
        marker in message for marker in (
            "cuda error", "cuda out of memory", "cuda driver", "cuda initialization",
            "cuda unavailable", "cuda is not available", "no kernel image",
            "invalid device function", "cublas", "cudnn", "not compiled with cuda",
            "found no nvidia driver", "cuda-capable device", "cuda backend",
        )
    )


def resize_minimax_image(image, width, height, algorithm):
    """Resize on CUDA when available; preserve input dtype and output device.

    CPU execution/recovery returns CPU output to avoid another GPU allocation.
    Lanczos uses the existing floating-point three-lobe kernel on both devices.
    """
    method = "lanczos3" if algorithm == "lanczos" else algorithm

    def resize(device):
        # The shared resizer also manages the source device's GPU cache. Move
        # to CPU first so recovery never calls cleanup hooks on a failing GPU.
        source = image.cpu() if device.type == "cpu" else image
        return resize_image_batch(
            source, width, height, method=method,
            antialias=algorithm == "lanczos", compute_device=device, chunk_size=1,
        )[0]

    compute_device = None
    result = None
    try:
        compute_device = resolve_resize_device(image.device)
        result = resize(compute_device)
        if compute_device.type == "cuda":
            # Surface asynchronous GPU errors before leaving the recovery block.
            torch.cuda.synchronize(compute_device)
            return result.to(image.device)
        return result
    except RuntimeError as error:
        if (compute_device is not None and compute_device.type != "cuda") or not _is_cuda_failure(error):
            raise
        warnings.warn(
            f"mAI prepare image for Minimax H3: CUDA failed; retrying on CPU. Reason: {error}",
            RuntimeWarning, stacklevel=2,
        )
    # Retry outside the handler so failed GPU scratch allocations can be freed.
    result = None
    return resize(torch.device("cpu"))
