"""Neutral-by-default image finishing with PyTorch, independent of ComfyUI."""

import math

import torch

from .image_blur import blur_image


# name: (default, minimum, maximum, step, tooltip)
POST_EFFECT_CONTROLS = {
    "contrast": (1.0, 0.0, 3.0, 0.01, "Multiplier around middle gray (0.5). 1 is neutral."),
    "lightness": (0.0, -1.0, 1.0, 0.01, "Add a brightness offset. 0 is neutral; positive is lighter."),
    "exposure": (0.0, -5.0, 5.0, 0.05, "Exposure in stops. +1 doubles RGB values; 0 is neutral."),
    "saturation": (1.0, 0.0, 3.0, 0.01, "Color multiplier. 0 is grayscale; 1 is neutral."),
    "gamma": (1.0, 0.1, 5.0, 0.01, "Midtone correction: RGB ** (1 / gamma). Above 1 brightens."),
    "blur": (0.0, 0.0, 32.0, 0.1, "Gaussian spread in native image pixels. 0 disables blur."),
    "sharpen": (0.0, 0.0, 3.0, 0.01, "Unsharp-mask amount with a fixed 1-pixel Gaussian spread. 0 disables."),
    "strength": (1.0, 0.0, 1.0, 0.01, "Blend processed color with the original. 0 bypasses all effects."),
}


def validate_post_effects(image, enabled=True, **controls):
    """Validate before selecting a GPU or allocating image buffers."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch tensor")
    if image.ndim != 4 or any(size <= 0 for size in image.shape):
        raise ValueError("image must have nonempty shape [B, H, W, C]")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError("image must have 1, 3 or 4 channels")
    if image.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("image must use float16, bfloat16, float32 or float64")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    unknown = controls.keys() - POST_EFFECT_CONTROLS.keys()
    if unknown:
        raise ValueError(f"Unknown post effect controls: {', '.join(sorted(unknown))}")
    settings = {}
    for name, (default, minimum, maximum, _step, _tooltip) in POST_EFFECT_CONTROLS.items():
        value = controls.get(name, default)
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}")
        settings[name] = float(value)
    return settings


def post_effects_bypassed(enabled, settings):
    return not enabled or settings["strength"] == 0 or all(
        settings[name] == spec[0]
        for name, spec in POST_EFFECT_CONTROLS.items() if name != "strength"
    )


@torch.no_grad()
def apply_image_post_effects(image, enabled=True, compute_device=None, output_device=None, **controls):
    """Process BHWC frames separately; preserve alpha, dimensions and dtype.

    Order: exposure, contrast, lightness, saturation, gamma, blur, sharpen,
    strength. Color is clamped before gamma and after spatial effects. The
    default output device matches the input; CPU fallback can request CPU output.
    Half inputs compute in float32, while double precision stays double.
    """
    settings = validate_post_effects(image, enabled, **controls)
    if post_effects_bypassed(enabled, settings):
        return image
    device = image.device if compute_device is None else torch.device(compute_device)
    destination = image.device if output_device is None else torch.device(output_device)
    dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    channels = min(3, image.shape[-1])
    result = torch.empty_like(image, device=destination)
    with torch.autocast(device_type=device.type, enabled=False):
        for index in range(image.shape[0]):
            source = image[index:index + 1, ..., :channels].to(device=device, dtype=dtype)
            # Out-of-place operations keep the input unmodified even on the same device.
            work = source * (2.0 ** settings["exposure"])
            work = (work - 0.5) * settings["contrast"] + 0.5 + settings["lightness"]
            if channels == 3 and settings["saturation"] != 1:
                gray = (work * work.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1, keepdim=True)
                work = gray + (work - gray) * settings["saturation"]
            work = work.clamp(0, 1)
            if settings["gamma"] != 1:
                work = work.pow(1.0 / settings["gamma"])
            if settings["blur"] > 0:
                work = blur_image(work, strength=settings["blur"])
            if settings["sharpen"] > 0:
                work = work + settings["sharpen"] * (work - blur_image(work, strength=1.0))
            work = work.clamp(0, 1)
            if settings["strength"] != 1:
                work = torch.lerp(source, work, settings["strength"]).clamp(0, 1)
            result[index:index + 1, ..., :channels] = work.to(device=destination, dtype=image.dtype)
            if channels == 3 and image.shape[-1] == 4:
                result[index:index + 1, ..., 3:] = image[index:index + 1, ..., 3:].to(destination)
    return result
