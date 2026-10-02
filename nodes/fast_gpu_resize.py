"""ComfyUI wrapper for batched, tensor-only resizing."""

import torch

try:
    from ..utils.resize_kernels import RESIZE_METHODS, RESIZE_MODES, resize_image_batch
except ImportError:
    from utils.resize_kernels import RESIZE_METHODS, RESIZE_MODES, resize_image_batch


def resolve_compute_device(image_device, device):
    if device == "auto":
        return image_device
    if device == "cpu":
        return torch.device("cpu")
    if device != "gpu":
        raise ValueError(f"Unknown device: {device}")
    if not torch.cuda.is_available():
        raise ValueError("GPU resizing requires an available CUDA device; choose auto or cpu.")
    # Import only during explicit GPU execution, never while loading the pack.
    try:
        from comfy.model_management import get_torch_device
    except ModuleNotFoundError as exc:
        if exc.name not in ("comfy", "comfy.model_management"):
            raise
        preferred = torch.device("cuda", torch.cuda.current_device())
    else:
        preferred = get_torch_device()
    if preferred.type != "cuda":
        raise ValueError("ComfyUI's preferred device is not CUDA; choose auto or cpu.")
    return preferred


class MAIFastGPUResize:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "method_used")
    DESCRIPTION = (
        "Batched torch resize with native, Lanczos and BC cubic kernels. "
        "Auto device keeps the input device; select gpu to move CPU images to CUDA. "
        "Fit pads black; fill crops from the center."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 1280, "min": 1, "max": 16384}),
            "height": ("INT", {"default": 720, "min": 1, "max": 16384}),
            "resize_mode": (list(RESIZE_MODES), {"default": "exact"}),
            "method": (list(RESIZE_METHODS), {"default": "auto"}),
            "antialias": ("BOOLEAN", {"default": True}),
            "multiple_of": ("INT", {"default": 1, "min": 1, "max": 16384, "step": 1,
                                    "tooltip": "Round output width and height to the nearest multiple of this number. 1 disables rounding; ties round upward."}),
            "device": (["auto", "gpu", "cpu"], {"default": "auto"}),
            "precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto"}),
            "chunk_size": ("INT", {"default": 0, "min": 0, "max": 1000000,
                                   "tooltip": "0: whole batch for native modes, automatic memory-bounded chunks for custom kernels. Positive: frames per chunk."}),
        }}

    def run(self, image, width=1280, height=720, resize_mode="exact", method="auto",
            antialias=True, multiple_of=1, device="auto", precision="auto", chunk_size=0):
        if not isinstance(image, torch.Tensor):
            raise ValueError("image must be a ComfyUI IMAGE tensor.")
        return resize_image_batch(
            image, width, height, resize_mode, method, antialias, multiple_of,
            resolve_compute_device(image.device, device), precision, chunk_size,
        )
