"""Native-pixel smart crops and their source-space mask snapshots."""

import math

import torch
import torch.nn.functional as F

from .mask_crop_geometry import calculate_crop_geometry, valid_crop_bounds


CONTEXT_TYPE = "MAI_MASK_CROP_CONTEXT"
CONTEXT_VERSION = 1


def validate_image(image, name="image"):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or min(image.shape) < 1:
        raise ValueError(f"{name} must be a non-empty IMAGE tensor [B, H, W, C]")
    if not image.is_floating_point():
        raise ValueError(f"{name} must have a floating point dtype")


def normalize_mask(mask, batch_size, height, width, device, dtype, name="mask"):
    """Broadcast only a singleton batch; never silently resize source masks."""
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{name} must be a MASK tensor [B, H, W] or [H, W]")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or tuple(mask.shape[1:]) != (height, width):
        raise ValueError(f"{name} must have spatial dimensions {width} x {height}")
    if mask.shape[0] not in (1, batch_size):
        raise ValueError(f"{name} batch must be 1 or match the image batch ({batch_size})")
    return mask.to(device=device, dtype=dtype).clamp(0, 1).expand(batch_size, -1, -1)


def resize_nchw(tensor, height, width, mode):
    """Antialiased interpolation in fp32 (fp64 stays fp64), on the same device."""
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    dtype = tensor.dtype
    work = tensor if dtype in (torch.float32, torch.float64) else tensor.float()
    return F.interpolate(work, size=(height, width), mode=mode,
                         align_corners=False, antialias=True).clamp(0, 1).to(dtype)


def find_mask_bbox(mask, threshold):
    """Transfer only four coordinates, avoiding an Nx2 list of active pixels."""
    active = mask > threshold
    rows = torch.where(active.any(dim=1))[0]
    if rows.numel() == 0:
        return None
    columns = torch.where(active.any(dim=0))[0]
    top, bottom, left, right = torch.stack(
        (rows[0], rows[-1] + 1, columns[0], columns[-1] + 1)
    ).tolist()
    return left, top, right, bottom


def extract_source_image(image, context):
    left, top, right, bottom = valid_crop_bounds(context)
    source = image[top:bottom, left:right].permute(2, 0, 1).unsqueeze(0)
    if context["was_padded"]:
        # Replicate works even for one-pixel images and arbitrarily large pads.
        source = F.pad(source, (context["pad_left"], context["pad_right"],
                                context["pad_top"], context["pad_bottom"]), mode="replicate")
    return source


def smart_crop(image, mask, output_width=1024, output_height=1024,
               mask_padding=0, mask_threshold=0.01, allow_upscale=False,
               edge_mode="shift"):
    validate_image(image)
    if (not isinstance(mask_threshold, (int, float)) or
            not math.isfinite(mask_threshold) or not 0 <= mask_threshold <= 1):
        raise ValueError("mask_threshold must be a finite number between 0 and 1")
    batch_size, height, width, _ = image.shape
    # Do not quantize a float32 mask before thresholding a half-precision image.
    mask_dtype = mask.dtype if isinstance(mask, torch.Tensor) and mask.is_floating_point() else torch.float32
    masks = normalize_mask(mask, batch_size, height, width, image.device, mask_dtype)
    full_crop_mask = masks.new_zeros((batch_size, height, width))
    images, cropped_masks, items = [], [], []
    for index in range(batch_size):
        context = calculate_crop_geometry(
            width, height, find_mask_bbox(masks[index], mask_threshold),
            output_width, output_height, mask_padding, edge_mode, allow_upscale,
        )
        left, top, right, bottom = valid_crop_bounds(context)
        full_crop_mask[index, top:bottom, left:right] = 1
        # Keep an independent native mask snapshot so interpolation cannot leak
        # edits outside the original semantic mask during default stitching.
        native_mask = masks[index, top:bottom, left:right].clone()
        context["source_mask"] = native_mask
        source_mask = F.pad(native_mask[None, None],
                            (context["pad_left"], context["pad_right"],
                             context["pad_top"], context["pad_bottom"]), value=0)
        source_image = extract_source_image(image[index], context)
        images.append(resize_nchw(source_image, output_height, output_width, "bicubic")
                      .squeeze(0).permute(1, 2, 0))
        cropped_masks.append(resize_nchw(source_mask, output_height, output_width, "bilinear")[0, 0])
        items.append(context)

    # Single images expose the metadata directly. Batches retain every item;
    # scalar output sockets and top-level geometry describe item zero.
    crop_context = dict(items[0])
    crop_context.update(type=CONTEXT_TYPE, version=CONTEXT_VERSION,
                        batch_size=batch_size, items=items)
    first = items[0]
    return (torch.stack(images), torch.stack(cropped_masks), full_crop_mask, crop_context,
            first["crop_x"], first["crop_y"], first["source_crop_width"],
            first["source_crop_height"], first["scale"])
