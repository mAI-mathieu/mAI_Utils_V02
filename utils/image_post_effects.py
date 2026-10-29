"""Neutral-by-default image finishing with PyTorch, independent of ComfyUI."""

import math

import torch
import torch.nn.functional as F

from .image_blur import blur_image


# name: (default, minimum, maximum, step, tooltip)
POST_EFFECT_CONTROLS = {
    "contrast": (1.0, 0.0, 3.0, 0.01, "Multiplier around middle gray (0.5). 1 is neutral."),
    "lightness": (0.0, -1.0, 1.0, 0.01, "Add a brightness offset. 0 is neutral; positive is lighter."),
    "exposure": (0.0, -5.0, 5.0, 0.05, "Exposure in stops. +1 doubles RGB values; 0 is neutral."),
    "saturation": (1.0, 0.0, 3.0, 0.01, "Color multiplier. 0 is grayscale; 1 is neutral."),
    "gamma": (1.0, 0.1, 5.0, 0.01, "Midtone correction: RGB ** (1 / gamma). Above 1 brightens."),
    "blur": (0.0, 0.0, 32.0, 0.1, "Gaussian spread in native image pixels. 0 disables blur."),
    "sharpen": (0.0, 0.0, 3.0, 0.01, "Unsharp-mask amount. Adjust sharpen_radius for the detail size. 0 disables."),
    "strength": (1.0, 0.0, 1.0, 0.01, "Blend processed color with the original. 0 bypasses all effects."),
}


# Kept separate so new widgets can be appended after all released widgets.
DETAIL_CONTROLS = {
    "texture": (0.0, 0.0, 2.0, 0.01, "Fine detail contrast between Gaussian spreads of 0.7 and 2 pixels, scaled by detail_scale."),
    "clarity": (0.0, 0.0, 1.0, 0.01, "Medium detail contrast using an 8-pixel Gaussian spread, scaled by detail_scale."),
    "detail_scale": (1.0, 0.25, 4.0, 0.05, "Multiplies texture, clarity and sharpening spreads in native output pixels. Increase for larger upscaled features."),
    "sharpen_radius": (1.0, 0.1, 8.0, 0.1, "Sharpening Gaussian spread in pixels, multiplied by detail_scale."),
    "detail_threshold": (0.0, 0.0, 0.1, 0.001, "Suppress detail differences below this normalized value in texture, clarity and sharpening. Helps avoid enhancing noise."),
    "halo_suppression": (0.0, 0.0, 1.0, 0.01, "Limit detail enhancement to nearby brightness/color extrema. 0 preserves legacy sharpening; 1 fully limits overshoot."),
}

REFINEMENT_PRESETS = {
    "Prepare for refinement - Low": {
        "texture": 0.2, "clarity": 0.12, "sharpen": 0.25,
        "detail_threshold": 0.01, "halo_suppression": 0.8,
    },
    "Prepare for refinement - Medium": {
        "texture": 0.45, "clarity": 0.25, "sharpen": 0.5,
        "detail_threshold": 0.007, "halo_suppression": 0.75,
    },
    "Prepare for refinement - High": {
        "texture": 0.8, "clarity": 0.4, "sharpen": 0.85,
        "detail_threshold": 0.005, "halo_suppression": 0.65,
    },
}
POST_EFFECT_PRESETS = ("Manual", *REFINEMENT_PRESETS)


def normalize_post_effects_preset(preset):
    # Keep API calls and saved workflows using the original label compatible.
    if preset == "Prepare for refinement":
        return "Prepare for refinement - Low"
    if preset not in POST_EFFECT_PRESETS:
        raise ValueError(f"Unknown post effects preset: {preset}")
    return preset


def prepare_effect_mask(mask, image_shape):
    """Validate source-size floating masks, broadcasting only singleton batches."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
        raise ValueError("effect_mask must have shape [H, W] or [B, H, W]")
    if mask.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("effect_mask must use a floating dtype")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[1:] != image_shape[1:3] or mask.shape[0] not in (1, image_shape[0]):
        raise ValueError("effect_mask must match image dimensions and have one mask or match the image batch")
    return mask


def validate_post_effects(image, enabled=True, preset="Manual", luminance_only=False, **controls):
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
    if not isinstance(luminance_only, bool):
        raise ValueError("luminance_only must be a boolean")
    preset = normalize_post_effects_preset(preset)
    specifications = {**POST_EFFECT_CONTROLS, **DETAIL_CONTROLS}
    unknown = controls.keys() - specifications.keys()
    if unknown:
        raise ValueError(f"Unknown post effect controls: {', '.join(sorted(unknown))}")
    settings = {}
    for name, (default, minimum, maximum, _step, _tooltip) in specifications.items():
        value = controls.get(name, default)
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be a finite number between {minimum} and {maximum}")
        settings[name] = float(value)
    settings["luminance_only"] = luminance_only
    if preset in REFINEMENT_PRESETS:
        # A predictable base before a low-denoise pass. Strength and detail_scale
        # remain user-controlled; the preset never changes resolution itself.
        settings.update({name: spec[0] for name, spec in POST_EFFECT_CONTROLS.items() if name != "strength"})
        settings.update(sharpen_radius=0.8, luminance_only=True, **REFINEMENT_PRESETS[preset])
    return settings


def post_effects_bypassed(enabled, settings):
    neutral = all(settings[name] == spec[0] for name, spec in POST_EFFECT_CONTROLS.items()
                  if name != "strength") and settings["texture"] == 0 and settings["clarity"] == 0
    return not enabled or settings["strength"] == 0 or neutral


def _enhance_detail(signal, detail, amount, threshold, halo_suppression):
    """Soft noise gate and local 3x3 overshoot limiter for one detail layer."""
    if threshold > 0:
        detail = detail.sign() * (detail.abs() - threshold).clamp_min(0)
    enhanced = signal + amount * detail
    if halo_suppression > 0:
        nchw = signal.movedim(-1, 1)
        padded = F.pad(nchw, (1, 1, 1, 1), mode="replicate")
        high = F.max_pool2d(padded, 3, stride=1).movedim(1, -1)
        low = -F.max_pool2d(-padded, 3, stride=1).movedim(1, -1)
        limited = torch.maximum(low, torch.minimum(high, enhanced))
        enhanced = torch.lerp(enhanced, limited, halo_suppression)
    return enhanced


def _apply_detail_effects(work, settings):
    """Texture then clarity then sharpening; optionally change luminance only."""
    luminance_only = settings["luminance_only"] and work.shape[-1] == 3
    signal = (work * work.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1, keepdim=True) if luminance_only else work
    original_signal = signal
    scale = settings["detail_scale"]
    if settings["texture"] > 0:
        detail = blur_image(signal, 0.7 * scale) - blur_image(signal, 2.0 * scale)
        signal = _enhance_detail(signal, detail, settings["texture"],
                                 settings["detail_threshold"], settings["halo_suppression"])
    if settings["clarity"] > 0:
        detail = signal - blur_image(signal, 8.0 * scale)
        signal = _enhance_detail(signal, detail, settings["clarity"],
                                 settings["detail_threshold"], settings["halo_suppression"])
    if settings["sharpen"] > 0:
        detail = signal - blur_image(signal, settings["sharpen_radius"] * scale)
        signal = _enhance_detail(signal, detail, settings["sharpen"],
                                 settings["detail_threshold"], settings["halo_suppression"])
    if luminance_only:
        # Add the same luminance change to every channel. Bound it to available
        # gamut headroom instead of clipping individual channels and shifting color.
        delta = signal - original_signal
        delta = torch.maximum(-work.amin(-1, keepdim=True),
                              torch.minimum(1 - work.amax(-1, keepdim=True), delta))
        return work + delta
    return signal


@torch.no_grad()
def apply_image_post_effects(image, enabled=True, compute_device=None, output_device=None,
                             effect_mask=None, preset="Manual", luminance_only=False, **controls):
    """Process BHWC frames separately; preserve alpha, dimensions and dtype.

    Order: exposure, contrast, lightness, saturation, gamma, blur, texture,
    clarity, sharpen, strength/mask. Color is clamped before gamma and after spatial effects. The
    default output device matches the input; CPU fallback can request CPU output.
    Half inputs compute in float32, while double precision stays double.
    """
    settings = validate_post_effects(image, enabled, preset, luminance_only, **controls)
    effect_mask = prepare_effect_mask(effect_mask, image.shape)
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
            if any(settings[name] > 0 for name in ("texture", "clarity", "sharpen")):
                work = _apply_detail_effects(work, settings)
            work = work.clamp(0, 1)
            if settings["strength"] != 1:
                work = torch.lerp(source, work, settings["strength"]).clamp(0, 1)
            if effect_mask is not None:
                mask_index = 0 if effect_mask.shape[0] == 1 else index
                mask = effect_mask[mask_index:mask_index + 1].unsqueeze(-1).to(device=device, dtype=dtype)
                if not torch.isfinite(mask).all().item():
                    raise ValueError("effect_mask must contain finite values")
                mask = mask.clamp(0, 1)
                work = torch.lerp(source, work, mask)
                # Exact protection includes out-of-range source values.
                work = torch.where(mask == 0, source, work)
            result[index:index + 1, ..., :channels] = work.to(device=destination, dtype=image.dtype)
            if channels == 3 and image.shape[-1] == 4:
                result[index:index + 1, ..., 3:] = image[index:index + 1, ..., 3:].to(destination)
    return result
