"""ComfyUI image finishing with automatic CUDA selection and CPU recovery."""

import sys
import warnings

import torch

try:
    from ..utils.image_post_effects import (
        POST_EFFECT_CONTROLS, apply_image_post_effects, post_effects_bypassed, validate_post_effects,
    )
except ImportError:
    from utils.image_post_effects import (
        POST_EFFECT_CONTROLS, apply_image_post_effects, post_effects_bypassed, validate_post_effects,
    )


def resolve_post_effects_device(image_device, device):
    if device not in ("auto", "gpu", "cpu"):
        raise ValueError(f"Unknown post effects device: {device}")
    if device == "cpu":
        return torch.device("cpu")
    # ComfyUI has already loaded model management during startup. Looking it up
    # avoids importing/initializing ComfyUI in standalone utility tests.
    management = sys.modules.get("comfy.model_management")
    preferred = torch.device(management.get_torch_device()) if management else None
    if preferred is not None and preferred.type != "cuda":
        if device == "gpu":
            raise ValueError("ComfyUI's selected device is not CUDA; choose auto or cpu.")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if device == "gpu":
            raise ValueError("GPU post effects require CUDA; choose auto for CPU fallback or cpu.")
        return torch.device("cpu")
    if preferred is not None:
        return preferred
    if image_device.type == "cuda":
        return image_device
    return torch.device("cuda", torch.cuda.current_device())


def _is_cuda_failure(error):
    message = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or any(marker in message for marker in (
        "cuda error", "cuda out of memory", "cuda driver", "cuda initialization",
        "cuda unavailable", "cuda is not available", "no kernel image", "invalid device function",
        "cublas_status", "cudnn_status", "cudnn error", "not compiled with cuda",
        "found no nvidia driver", "cuda-capable device", "cuda backend", "'cuda' backend",
    ))


class MAIImagePostEffects:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Contrast, lightness, exposure, saturation, gamma, Gaussian blur and sharpening. "
        "Defaults preserve the image exactly. Auto uses ComfyUI's selected CUDA GPU "
        "and retries hardware failures on CPU. Alpha stays unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        controls = {
            name: ("FLOAT", {"default": default, "min": minimum, "max": maximum,
                             "step": step, "tooltip": tooltip})
            for name, (default, minimum, maximum, step, tooltip) in POST_EFFECT_CONTROLS.items()
        }
        return {"required": {
            "image": ("IMAGE",),
            "enabled": ("BOOLEAN", {"default": True}),
            **controls,
            "device": (["auto", "gpu", "cpu"], {"default": "auto",
                       "tooltip": "Auto accelerates with CUDA and falls back to CPU. GPU requires CUDA. CPU forces CPU output."}),
        }}

    def run(self, image, enabled=True, device="auto", **controls):
        settings = validate_post_effects(image, enabled, **controls)
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"Unknown post effects device: {device}")
        if post_effects_bypassed(enabled, settings):
            return (image,)
        compute_device = None
        try:
            compute_device = resolve_post_effects_device(image.device, device)
            output_device = "cpu" if compute_device.type == "cpu" else image.device
            return (apply_image_post_effects(
                image, compute_device=compute_device, output_device=output_device, **settings,
            ),)
        except RuntimeError as exc:
            if device != "auto" or (
                compute_device is not None and compute_device.type != "cuda"
            ) or not _is_cuda_failure(exc):
                raise
            warnings.warn(
                f"mAI image post effects: CUDA failed; retrying on CPU. Reason: {exc}",
                RuntimeWarning, stacklevel=2,
            )
        # Retry outside the handler to release failed GPU scratch tensors.
        # Keep CPU output so fallback never allocates on a failing GPU again.
        return (apply_image_post_effects(
            image, compute_device="cpu", output_device="cpu", **settings,
        ),)
