"""ComfyUI wrapper and device policy for Gaussian image blur."""

import warnings

import torch

try:
    from ..utils.image_blur import blur_image
except ImportError:
    from utils.image_blur import blur_image


def resolve_blur_device(image_device, device):
    if device == "cpu":
        return torch.device("cpu")
    if device not in ("auto", "gpu"):
        raise ValueError(f"Unknown blur device: {device}")
    if not torch.cuda.is_available():
        if device == "gpu":
            raise ValueError("GPU blur requires CUDA; choose auto for CPU fallback or cpu.")
        return torch.device("cpu")
    if device == "auto" and image_device.type == "cuda":
        return image_device
    # Respect ComfyUI's selected GPU, including multi-GPU setups. Import only
    # during execution so loading the pack never initializes CUDA.
    try:
        from comfy.model_management import get_torch_device
    except ModuleNotFoundError as exc:
        if exc.name not in ("comfy", "comfy.model_management"):
            raise
        return torch.device("cuda", torch.cuda.current_device())
    preferred = torch.device(get_torch_device())
    if preferred.type != "cuda":
        if device == "gpu":
            raise ValueError("ComfyUI's selected device is not CUDA; choose auto or cpu.")
        return torch.device("cpu")
    return preferred


class MAIImageBlur:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {"default": 8, "min": 0, "max": 128}),
                "sigma": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 128.0, "step": 0.1}),
                "device": (["auto", "cpu", "gpu"], {"default": "auto"}),
            },
        }

    def run(self, image, radius=8, sigma=3.0, device="auto"):
        compute_device = None
        try:
            compute_device = resolve_blur_device(image.device, device)
            return (blur_image(image, radius, sigma, compute_device),)
        except RuntimeError as exc:
            if device != "auto" or (
                compute_device is not None and compute_device.type != "cuda"
            ):
                raise
            warnings.warn(
                f"mAI image blur: CUDA failed; retrying on CPU. Reason: {exc}",
                RuntimeWarning, stacklevel=2,
            )
        # Leave the exception handler first so its traceback releases temporary
        # CUDA tensors before allocating CPU work. Never hide CPU errors.
        return (blur_image(image, radius, sigma, torch.device("cpu")),)
