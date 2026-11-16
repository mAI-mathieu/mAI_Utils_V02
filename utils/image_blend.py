"""Blend aligned ComfyUI image batches without a ComfyUI runtime dependency."""

from numbers import Real

import torch

from .image_blur import blur_image


FREQUENCY_MODES = ("high_frequency", "low_frequency")
BLEND_MODES = ("normal", "multiply", "screen", "overlay", "soft_light", "difference", *FREQUENCY_MODES)
DEFAULT_DETAIL_SCALE = 2.0


def parse_blend_percentages(blend_list):
    """Read comma- or newline-separated percentages in their original order."""
    if not isinstance(blend_list, str):
        raise TypeError("blend_list must be text containing percentages.")

    percentages = []
    for entry in blend_list.replace(",", "\n").splitlines():
        entry = entry.strip()
        if not entry:
            continue
        try:
            percent = float(entry.removesuffix("%"))
        except ValueError as exc:
            raise ValueError(f"Invalid blend percentage {entry!r}.") from exc
        if not 0 <= percent <= 100:
            raise ValueError(f"Blend percentage {entry!r} must be from 0 to 100.")
        percentages.append(percent)

    if not percentages:
        raise ValueError("blend_list must contain at least one percentage, such as 50%.")
    return percentages


def _apply_blend_mode(bottom, top, mode):
    if mode == "normal":
        return top
    if mode == "multiply":
        return bottom * top
    if mode == "screen":
        return 1 - (1 - bottom) * (1 - top)
    if mode == "overlay":
        return torch.where(bottom <= 0.5, 2 * bottom * top, 1 - 2 * (1 - bottom) * (1 - top))
    if mode == "soft_light":
        curve = torch.where(bottom <= 0.25, ((16 * bottom - 12) * bottom + 4) * bottom, bottom.sqrt())
        return torch.where(
            top <= 0.5,
            bottom - (1 - 2 * top) * bottom * (1 - bottom),
            bottom + (2 * top - 1) * (curve - bottom),
        )
    if mode == "difference":
        return (bottom - top).abs()
    raise ValueError(f"Unknown blend mode: {mode}")


def _blend_frequency_frame(top, base, fraction, mode, detail_scale):
    difference = top - base
    if mode == "high_frequency" and base.shape[-1] == 3:
        luma_weights = difference.new_tensor([0.2126, 0.7152, 0.0722])
        difference = (difference * luma_weights).sum(dim=-1, keepdim=True)
    low_difference = blur_image(difference, strength=detail_scale)

    if mode == "high_frequency":
        adjustment = difference.sub_(low_difference).mul_(fraction)
        # A shared, gamut-limited RGB offset preserves the base chroma at saturated pixels.
        adjustment.clamp_(min=-base.amin(dim=-1, keepdim=True), max=1 - base.amax(dim=-1, keepdim=True))
    else:
        adjustment = low_difference.mul_(fraction)
    return (base + adjustment).clamp_(0, 1)


@torch.no_grad()
def _blend_frames(top, bottom, percentages, mode, detail_scale, batch_size, device, output_device):
    top = top.expand(batch_size, -1, -1, -1)
    bottom = bottom.expand(batch_size, -1, -1, -1)
    result = bottom.new_empty(bottom.shape, device=output_device)
    if len(percentages) == 1:
        percentages = percentages * batch_size

    frequency = mode in FREQUENCY_MODES
    channels = min(bottom.shape[-1], 3) if frequency else bottom.shape[-1]
    dtype = torch.float32 if frequency and bottom.dtype != torch.float64 else bottom.dtype
    weights = None if frequency else bottom.new_tensor([p / 100 for p in percentages], device=device)

    # Compute and transfer intermediate tensors one frame at a time for video batches.
    for index, percent in enumerate(percentages):
        output = result[index:index + 1, ..., :channels]
        base = bottom[index:index + 1, ..., :channels]
        if frequency and percent == 0:
            output.copy_(base)
            continue
        base = base.to(device=device, dtype=dtype)
        model = top[index:index + 1, ..., :channels].to(device=device, dtype=dtype)
        if frequency:
            blended = _blend_frequency_frame(model, base, percent / 100, mode, detail_scale)
        else:
            blended = torch.lerp(base, _apply_blend_mode(base, model, mode), weights[index])
        output.copy_(blended)

    if frequency and bottom.shape[-1] == 4:
        result[..., 3:4].copy_(bottom[..., 3:4])
    return result


def blend_image_batches(
    image_top, image_bottom, blend_percentage=50.0, blend_mode="normal", *,
    blend_list="", detail_scale=DEFAULT_DETAIL_SCALE, compute_device=None, output_device=None,
):
    """Interpolate from the bottom image to the selected mode's result."""
    if (isinstance(blend_percentage, bool) or not isinstance(blend_percentage, Real)
            or not 0 <= blend_percentage <= 100):
        raise ValueError("blend_percentage must be a number from 0 to 100.")
    percentages = parse_blend_percentages(blend_list) if blend_list else [blend_percentage]
    for name, image in (("image_top", image_top), ("image_bottom", image_bottom)):
        if (not isinstance(image, torch.Tensor) or image.ndim != 4
                or min(image.shape) < 1 or not image.is_floating_point()):
            raise ValueError(f"{name} must be a nonempty floating-point IMAGE batch [B, H, W, C].")

    if image_top.shape[1:] != image_bottom.shape[1:]:
        raise ValueError("image_top and image_bottom must have matching height, width, and channels.")

    counts = (image_top.shape[0], image_bottom.shape[0], len(percentages))
    batch_size = max(counts)
    if any(count not in (1, batch_size) for count in counts):
        raise ValueError(
            "Each input must have one item or the same number of items. "
            f"Got {counts[0]} top images, {counts[1]} bottom images, and {counts[2]} blend percentages."
        )

    if blend_mode in FREQUENCY_MODES:
        if (isinstance(detail_scale, bool) or not isinstance(detail_scale, Real)
                or not 0.1 <= detail_scale <= 32):
            raise ValueError("detail_scale must be a Gaussian spread from 0.1 to 32 output pixels.")
        if image_bottom.shape[-1] not in (1, 3, 4):
            raise ValueError("Frequency blending requires grayscale, RGB, or RGBA images.")
        detail_scale = float(detail_scale)

    device = image_bottom.device if compute_device is None else torch.device(compute_device)
    output_device = image_bottom.device if output_device is None else torch.device(output_device)
    if (blend_mode in FREQUENCY_MODES or device != image_bottom.device
            or device != image_top.device or device != output_device):
        return _blend_frames(
            image_top, image_bottom, percentages, blend_mode, detail_scale, batch_size, device, output_device,
        )

    image_top = image_top.to(image_bottom)
    weights = image_bottom.new_tensor([percent / 100 for percent in percentages])
    blended = _apply_blend_mode(image_bottom, image_top, blend_mode)
    return torch.lerp(image_bottom, blended, weights.reshape(-1, 1, 1, 1))
