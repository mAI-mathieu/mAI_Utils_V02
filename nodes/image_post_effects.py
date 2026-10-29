"""ComfyUI image finishing with automatic CUDA selection and CPU recovery."""

import sys
import warnings

import torch

try:
    from ..utils.image_post_effects import (
        DETAIL_CONTROLS, POST_EFFECT_CONTROLS, POST_EFFECT_PRESETS, apply_image_post_effects,
        normalize_post_effects_preset, post_effects_bypassed, prepare_effect_mask, validate_post_effects,
    )
except ImportError:
    from utils.image_post_effects import (
        DETAIL_CONTROLS, POST_EFFECT_CONTROLS, POST_EFFECT_PRESETS, apply_image_post_effects,
        normalize_post_effects_preset, post_effects_bypassed, prepare_effect_mask, validate_post_effects,
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
        "Tone, color, blur, texture, clarity and controlled sharpening. Prepare for refinement "
        "offers Low, Medium and High luminance detail before VAE encoding and low-denoise sampling, including after upscaling. "
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
        detail_controls = {
            name: ("FLOAT", {"default": default, "min": minimum, "max": maximum,
                             "step": step, "tooltip": tooltip})
            for name, (default, minimum, maximum, step, tooltip) in DETAIL_CONTROLS.items()
        }
        return {"required": {
            "image": ("IMAGE",),
            "enabled": ("BOOLEAN", {"default": True}),
            **controls,
            "device": (["auto", "gpu", "cpu"], {"default": "auto",
                       "tooltip": "Auto accelerates with CUDA and falls back to CPU. GPU requires CUDA. CPU forces CPU output."}),
        }, "optional": {
            "preset": (list(POST_EFFECT_PRESETS), {"default": "Manual", "tooltip":
                       "Refinement Low is the original recipe; Medium and High increase detail enhancement. Presets override tone/color/blur and detail sliders. Strength, detail_scale, enabled and mask still apply."}),
            **detail_controls,
            "luminance_only": ("BOOLEAN", {"default": False, "tooltip":
                               "Process texture, clarity and sharpening in luminance, with gamut protection. All refinement presets enable this automatically."}),
            "effect_mask": ("MASK", {"tooltip":
                            "Optional image-size effect mask. Black keeps original pixels exactly, white applies all effects. One mask broadcasts across the batch."}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, preset="Manual"):
        # Accept the released label without adding a duplicate dropdown entry.
        try:
            normalize_post_effects_preset(preset)
        except ValueError as exc:
            return str(exc)
        return True

    def run(self, image, enabled=True, device="auto", effect_mask=None, preset="Manual",
            luminance_only=False, **controls):
        settings = validate_post_effects(image, enabled, preset, luminance_only, **controls)
        effect_mask = prepare_effect_mask(effect_mask, image.shape)
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"Unknown post effects device: {device}")
        if post_effects_bypassed(enabled, settings):
            return (image,)
        compute_device = None
        try:
            compute_device = resolve_post_effects_device(image.device, device)
            output_device = "cpu" if compute_device.type == "cpu" else image.device
            return (apply_image_post_effects(
                image, compute_device=compute_device, output_device=output_device,
                effect_mask=effect_mask, **settings,
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
            image, compute_device="cpu", output_device="cpu", effect_mask=effect_mask, **settings,
        ),)
