"""Record resize padding geometry and recut processed tensors without resampling."""

from numbers import Integral

import torch

from .resize_kernels import calculate_target_size
from .resize_masks import prepare_resize_mask


def create_resize_stitcher(source_width, source_height, width, height,
                           resize_mode="exact", multiple_of=1):
    """Save the exact content rectangle; modes without padding use the full canvas."""
    geometry = calculate_target_size(
        source_width, source_height, width, height, resize_mode, multiple_of,
    )
    content_width, content_height = geometry.width, geometry.height
    if resize_mode in ("keep_aspect_fit", "pad_to_multiple", "pad_to_multiple_edge"):
        content_width, content_height = geometry.resize_width, geometry.resize_height
    return {
        "version": 1,
        "canvas_width": geometry.width,
        "canvas_height": geometry.height,
        "x": (geometry.width - content_width) // 2,
        "y": (geometry.height - content_height) // 2,
        "width": content_width,
        "height": content_height,
    }


def validate_resize_stitcher(stitcher, canvas_width, canvas_height):
    """Reject incompatible metadata or changed canvases instead of guessing offsets."""
    if not isinstance(stitcher, dict) or stitcher.get("version") != 1:
        raise ValueError("stitcher must come from mAI Fast GPU Resize (version 1).")
    for name in ("canvas_width", "canvas_height", "x", "y", "width", "height"):
        value = stitcher.get(name)
        minimum = 0 if name in ("x", "y") else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"stitcher {name} must be an integer >= {minimum}.")
    x, y, width, height = (stitcher[name] for name in ("x", "y", "width", "height"))
    if x + width > stitcher["canvas_width"] or y + height > stitcher["canvas_height"]:
        raise ValueError("stitcher content rectangle exceeds its saved canvas.")
    if (canvas_width, canvas_height) != (stitcher["canvas_width"], stitcher["canvas_height"]):
        raise ValueError(
            f"Image size {canvas_width}x{canvas_height} does not match stitcher canvas "
            f"{stitcher['canvas_width']}x{stitcher['canvas_height']}. "
            "Keep the padded image dimensions unchanged before mAI stitch."
        )
    return x, y, width, height


def recut_resize_padding(image, stitcher, mask=None):
    """Crop all frames and an optional mask using the saved rectangle, preserving dtype/device."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or any(d < 1 for d in image.shape):
        raise ValueError("image must be a non-empty torch tensor [batch, height, width, channels].")
    if image.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("image must use float32, float16, bfloat16 or float64 pixels.")
    x, y, width, height = validate_resize_stitcher(stitcher, image.shape[2], image.shape[1])
    mask = prepare_resize_mask(mask, image.shape)
    result = image[:, y:y + height, x:x + width]
    output_mask = (result.new_zeros(result.shape[:3]) if mask is None else
                   mask[:, y:y + height, x:x + width].to(device=image.device))
    return result, width, height, output_mask
