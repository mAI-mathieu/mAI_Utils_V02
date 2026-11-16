"""Blend aligned ComfyUI image batches without a ComfyUI runtime dependency."""

from numbers import Real

import torch


BLEND_MODES = ("normal", "multiply", "screen", "overlay", "soft_light", "difference")


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


def blend_image_batches(image_top, image_bottom, blend_percentage=50.0, blend_mode="normal", *, blend_list=""):
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

    image_top = image_top.to(image_bottom)
    weights = image_bottom.new_tensor([percent / 100 for percent in percentages])
    blended = _apply_blend_mode(image_bottom, image_top, blend_mode)
    return torch.lerp(image_bottom, blended, weights.reshape(-1, 1, 1, 1))
