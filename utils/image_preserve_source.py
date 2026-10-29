"""Detect substantial edits and composite them over an untouched source."""

import math

import torch
import torch.nn.functional as F

from .image_blur import blur_image
from .image_edit_detection import detect_image_edits
from .masked_image_composite import BLEND_MODES, composite_with_source_mask


def _number(name, value, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= maximum):
        raise ValueError(f"{name} must be a finite number between 0 and {maximum}")


def _prepare_images(original_image, edited_image, edit_mask):
    for name, image in (("original_image", original_image), ("edited_image", edited_image)):
        if (not isinstance(image, torch.Tensor) or image.ndim != 4
                or min(image.shape) < 1 or image.shape[-1] not in (3, 4)
                or not image.is_floating_point()):
            raise ValueError(f"{name} must be a non-empty floating RGB/RGBA image [B,H,W,C]")
        if not torch.isfinite(image).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if original_image.shape[1:] != edited_image.shape[1:]:
        raise ValueError("original_image and edited_image dimensions and channels must match")
    dtype = torch.float64 if original_image.dtype == torch.float64 else torch.float32
    if edit_mask is None:
        edit_mask = torch.ones((1, *original_image.shape[1:3]),
                               device=original_image.device, dtype=dtype)
    else:
        if not isinstance(edit_mask, torch.Tensor):
            raise TypeError("edit_mask must be a MASK tensor [H,W] or [B,H,W]")
        if edit_mask.ndim == 2:
            edit_mask = edit_mask.unsqueeze(0)
        if (edit_mask.ndim != 3 or min(edit_mask.shape) < 1
                or edit_mask.shape[1:] != original_image.shape[1:3]):
            raise ValueError("edit_mask must have shape [H,W] or [B,H,W] matching the images")
        if not torch.isfinite(edit_mask).all().item():
            raise ValueError("edit_mask must contain only finite values")
        edit_mask = edit_mask.to(device=original_image.device, dtype=dtype).clamp(0, 1)
    batch = max(original_image.shape[0], edited_image.shape[0], edit_mask.shape[0])
    if any(t.shape[0] not in (1, batch) for t in (original_image, edited_image, edit_mask)):
        raise ValueError("Image and edit_mask batch sizes must match unless a batch is 1")
    original = original_image.expand(batch, -1, -1, -1)
    edited = edited_image.to(device=original.device, dtype=original.dtype).expand(batch, -1, -1, -1)
    return original, edited, edit_mask.expand(batch, -1, -1), dtype


@torch.no_grad()
def preserve_source_image(
    original_image, edited_image, difference_threshold=0.08, transition_width=0.04,
    comparison_blur=2.0, detail_sensitivity=0.25, mask_expand=2, mask_blur=2.0,
    blend_mode="multiband", multiband_levels=5, edit_mask=None, mode="manual",
):
    """Return image and final selection mask (white selects the edit).

    Compare RGB both before and after a Gaussian blur. detail_sensitivity
    interpolates between broad changes (0) and raw pixel differences (1).
    Threshold with a smooth transition, expand, feather, then restrict by the
    optional edit_mask. The shared compositor protects exact black/white mask
    endpoints and applies seam/pyramid work only inside the gray band.
    Auto instead derives coherent regions and blending scales per image pair;
    manual tuning parameters are ignored in that mode.
    """
    if mode not in ("manual", "auto"):
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "auto":
        original, edited, allowed, dtype = _prepare_images(original_image, edited_image, edit_mask)
        result = torch.empty_like(original)
        masks = torch.empty(allowed.shape, device=original.device, dtype=dtype)
        for index in range(original.shape[0]):
            mask, settings = detect_image_edits(
                original[index:index + 1, ..., :3], edited[index:index + 1, ..., :3],
                allowed[index:index + 1],
            )
            masks[index] = mask[0]
            result[index:index + 1] = composite_with_source_mask(
                original[index:index + 1], edited[index:index + 1], mask,
                "content_aware_seam_multiband", settings.multiband_levels,
            )
        return result, masks
    for name, value, maximum in (
        ("difference_threshold", difference_threshold, 1),
        ("transition_width", transition_width, 1),
        ("comparison_blur", comparison_blur, 32),
        ("detail_sensitivity", detail_sensitivity, 1),
        ("mask_blur", mask_blur, 32),
    ):
        _number(name, value, maximum)
    if (isinstance(mask_expand, bool) or not isinstance(mask_expand, int)
            or not 0 <= mask_expand <= 128):
        raise ValueError("mask_expand must be an integer between 0 and 128")
    if blend_mode not in BLEND_MODES:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")
    if (isinstance(multiband_levels, bool) or not isinstance(multiband_levels, int)
            or not 1 <= multiband_levels <= 10):
        raise ValueError("multiband_levels must be an integer between 1 and 10")

    original, edited, allowed, dtype = _prepare_images(original_image, edited_image, edit_mask)
    masks = torch.empty(allowed.shape, device=original.device, dtype=dtype)
    # Process detection per image to limit temporary memory on large batches.
    with torch.autocast(device_type=original.device.type, enabled=False):
        for index in range(original.shape[0]):
            source_rgb = original[index:index + 1, ..., :3].to(dtype)
            edited_rgb = edited[index:index + 1, ..., :3].to(dtype)
            raw = (source_rgb - edited_rgb).abs().mean(-1)
            if comparison_blur > 0 and detail_sensitivity < 1:
                broad = (blur_image(source_rgb, comparison_blur)
                         - blur_image(edited_rgb, comparison_blur)).abs().mean(-1)
                difference = torch.lerp(broad, raw, detail_sensitivity)
            else:
                difference = raw
            width = min(transition_width, 1.0 - difference_threshold)
            if width > 0:
                mask = ((difference - difference_threshold) / width).clamp(0, 1)
                mask = mask.square() * (3 - 2 * mask)
            else:
                mask = (difference > difference_threshold).to(dtype)
            # Do not let forbidden edits seed expansion into an allowed region.
            mask = mask * (allowed[index:index + 1] > 0)
            if mask_expand:
                values = mask[:, None]
                kernel = 2 * mask_expand + 1
                values = F.max_pool2d(values, (1, kernel), stride=1, padding=(0, mask_expand))
                mask = F.max_pool2d(values, (kernel, 1), stride=1, padding=(mask_expand, 0))[:, 0]
            if mask_blur > 0:
                mask = blur_image(mask[..., None], mask_blur)[..., 0]
            # Restrict after spatial processing so black protection stays exact.
            masks[index] = (mask[0] * allowed[index]).clamp(0, 1)
    result = composite_with_source_mask(original, edited, masks, blend_mode, multiband_levels)
    return result, masks
