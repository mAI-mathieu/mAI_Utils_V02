"""Reverse smart-crop geometry and composite only valid source pixels."""

import torch
import torch.nn.functional as F

from .mask_crop_geometry import _positive_integer, valid_crop_bounds
from .mask_smart_crop import (
    CONTEXT_TYPE, CONTEXT_VERSION, normalize_mask, resize_nchw, validate_image,
)


COMPOSITE_MODES = ("full_crop", "mask_only", "mask_feather")


def validate_context(context, original_image, generated_image):
    if not isinstance(context, dict) or context.get("type") != CONTEXT_TYPE:
        raise ValueError("crop_context must come from mAI Mask Smart Crop")
    if context.get("version") != CONTEXT_VERSION:
        raise ValueError("Unsupported smart crop context version")
    items = context.get("items")
    batch_size, height, width, channels = original_image.shape
    if (not isinstance(items, list) or len(items) != batch_size or
            context.get("batch_size") != batch_size or generated_image.shape[0] != batch_size):
        raise ValueError("original_image, generated_image and crop_context batches must match")
    if generated_image.shape[-1] != channels:
        raise ValueError("generated_image channels must match original_image")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("crop_context contains an invalid batch item")
        for key in ("original_width", "original_height", "source_crop_width",
                    "source_crop_height", "output_width", "output_height"):
            _positive_integer(item.get(key), f"crop_context {key}")
        for key in ("crop_x", "crop_y"):
            if isinstance(item.get(key), bool) or not isinstance(item.get(key), int):
                raise ValueError(f"crop_context {key} must be an integer")
        if (item["original_width"], item["original_height"]) != (width, height):
            raise ValueError("original_image dimensions must match crop_context")
        if tuple(generated_image.shape[1:3]) != (item["output_height"], item["output_width"]):
            raise ValueError("generated_image dimensions must match the Smart Crop output")
        if item["source_crop_width"] * item["output_height"] != item["source_crop_height"] * item["output_width"]:
            raise ValueError("crop_context source/output aspect ratios must match")
        left, top, right, bottom = valid_crop_bounds(item)
        if right <= left or bottom <= top:
            raise ValueError("crop_context has no valid source image area")
        if item.get("source_mask") is not None:
            saved = item["source_mask"]
            if not isinstance(saved, torch.Tensor) or tuple(saved.shape) != (bottom - top, right - left):
                raise ValueError("crop_context source_mask does not match its valid source area")
    return items


def feather_mask(mask, radius):
    """Gaussian feather with a finite source-pixel radius, clipped to the crop."""
    if radius == 0:
        return mask
    dtype = mask.dtype
    work = mask[None, None].double() if dtype == torch.float64 else mask[None, None].float()
    offsets = torch.arange(-radius, radius + 1, device=mask.device, dtype=work.dtype)
    kernel = torch.exp(-0.5 * (offsets / max(radius / 3, 0.5)).square())
    kernel /= kernel.sum()
    work = F.conv2d(F.pad(work, (radius, radius, 0, 0), mode="replicate"), kernel[None, None, None, :])
    work = F.conv2d(F.pad(work, (0, 0, radius, radius), mode="replicate"), kernel[None, None, :, None])
    return work[0, 0].clamp(0, 1).to(dtype)


def smart_stitch(original_image, generated_image, crop_context, mask=None,
                 feather=0, composite_mode="mask_feather"):
    validate_image(original_image, "original_image")
    validate_image(generated_image, "generated_image")
    _positive_integer(feather, "feather", minimum=0)
    if composite_mode not in COMPOSITE_MODES:
        raise ValueError(f"Unknown composite_mode: {composite_mode}")
    items = validate_context(crop_context, original_image, generated_image)
    generated = generated_image.to(device=original_image.device, dtype=original_image.dtype)
    masks = None
    if mask is not None:
        masks = normalize_mask(mask, original_image.shape[0], generated.shape[1],
                               generated.shape[2], original_image.device,
                               original_image.dtype, name="stitch mask")
    result = original_image.clone()
    for index, item in enumerate(items):
        left, top, right, bottom = valid_crop_bounds(item)
        source_width, source_height = item["source_crop_width"], item["source_crop_height"]
        restored = resize_nchw(generated[index].permute(2, 0, 1)[None],
                               source_height, source_width, "bicubic")[0].permute(1, 2, 0)
        offset_x, offset_y = left - item["crop_x"], top - item["crop_y"]
        valid_image = restored[offset_y:offset_y + bottom - top, offset_x:offset_x + right - left]
        if composite_mode == "full_crop":
            result[index, top:bottom, left:right] = valid_image
            continue
        if masks is not None:
            restored_mask = resize_nchw(masks[index][None, None], source_height, source_width, "bilinear")[0, 0]
            alpha = restored_mask[offset_y:offset_y + bottom - top, offset_x:offset_x + right - left]
        elif item.get("source_mask") is not None:
            alpha = item["source_mask"].to(device=result.device, dtype=result.dtype).clamp(0, 1)
        else:
            alpha = result.new_ones((bottom - top, right - left))
        if composite_mode == "mask_feather":
            alpha = feather_mask(alpha, feather)
        original = original_image[index, top:bottom, left:right]
        alpha = alpha.unsqueeze(-1)
        blended = original + (valid_image - original) * alpha
        # Preserve exact endpoints and exact native round trips, even for soft masks.
        blended = torch.where(alpha == 0, original, torch.where(alpha == 1, valid_image, blended))
        result[index, top:bottom, left:right] = blended
    return result
