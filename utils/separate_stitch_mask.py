"""Pure tensor and geometry helpers for a separate CropAndStitch blend mask."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def normalize_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a ComfyUI mask to float BHW form and clamp it to 0..1."""

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"{name} must have shape [H,W] or [B,H,W], got {tuple(mask.shape)}")
    return mask.float().clamp(0.0, 1.0)


def broadcast_inputs(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    optional_context_mask: torch.Tensor | None,
    stitch_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate spatial sizes and apply ComfyUI's batch-of-one broadcasting rule."""

    if image.ndim != 4:
        raise ValueError(f"image must have shape [B,H,W,C], got {tuple(image.shape)}")

    image_height, image_width = image.shape[1:3]
    normalized = {
        "mask": normalize_mask(mask, "mask") if mask is not None else None,
        "optional_context_mask": (
            normalize_mask(optional_context_mask, "optional_context_mask")
            if optional_context_mask is not None
            else None
        ),
        "stitch_mask": normalize_mask(stitch_mask, "stitch_mask"),
    }
    normalized = {
        name: value.to(image.device) if value is not None else None
        for name, value in normalized.items()
    }
    for name, value in normalized.items():
        if value is not None and value.shape[1:] != (image_height, image_width):
            raise ValueError(
                f"{name} dimensions must match image dimensions. "
                f"Expected {(image_height, image_width)}, got {tuple(value.shape[1:])}"
            )

    batch_size = max(
        image.shape[0],
        *(value.shape[0] for value in normalized.values() if value is not None),
    )

    def expand(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.shape[0] not in (1, batch_size):
            raise ValueError(
                f"{name} batch size must be 1 or {batch_size}, got {value.shape[0]}"
            )
        return value if value.shape[0] == batch_size else value.expand(batch_size, *value.shape[1:]).clone()

    image = expand(image, "image")
    render_mask = normalized["mask"]
    if render_mask is None:
        render_mask = torch.zeros(
            (1, image_height, image_width), device=image.device, dtype=image.dtype
        )
    context_mask = normalized["optional_context_mask"]
    if context_mask is None:
        context_mask = torch.zeros(
            (1, image_height, image_width), device=image.device, dtype=image.dtype
        )

    return (
        image,
        expand(render_mask, "mask"),
        expand(context_mask, "optional_context_mask"),
        expand(normalized["stitch_mask"], "stitch_mask"),
    )


def preresize_dimensions(
    width: int,
    height: int,
    mode: str,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int, str | None]:
    """Mirror the installed upstream v3 pre-resize dimension rules."""

    if mode == "ensure minimum resolution":
        if width >= min_width and height >= min_height:
            return width, height, None
        scale = max(min_width / width, min_height / height)
        return math.ceil(width * scale), math.ceil(height * scale), "bilinear"

    if mode == "ensure maximum resolution":
        if width <= max_width and height <= max_height:
            return width, height, None
        scale = min(max_width / width, max_height / height)
        return int(width * scale), int(height * scale), "nearest"

    if mode == "ensure minimum and maximum resolution":
        if min_width <= width <= max_width and min_height <= height <= max_height:
            return width, height, None
        scale_min = max(min_width / width, min_height / height)
        scale_max = min(max_width / width, max_height / height)
        if scale_min > 1 and scale_max < 1:
            raise ValueError(
                "Cannot meet both minimum and maximum resolution requirements "
                "with aspect ratio preservation."
            )
        scale = scale_min if scale_min > 1 else scale_max
        if scale >= 1.0:
            return math.ceil(width * scale), math.ceil(height * scale), "nearest"
        return int(width * scale), int(height * scale), "nearest"

    raise ValueError(f"Unknown preresize mode: {mode}")


def extend_mask(
    mask: torch.Tensor,
    up_factor: float,
    down_factor: float,
    left_factor: float,
    right_factor: float,
) -> torch.Tensor:
    """Apply upstream outpainting placement while leaving new blend area transparent."""

    batch, height, width = mask.shape
    # Keep the operation order identical to upstream to avoid one-pixel float
    # rounding differences at canvas boundaries.
    new_height = int(height * (1.0 + up_factor - 1.0 + down_factor - 1.0))
    new_width = int(width * (1.0 + left_factor - 1.0 + right_factor - 1.0))
    if new_height < 0 or new_width < 0:
        raise ValueError(
            f"Outpainting factors produce an invalid mask size {new_width}x{new_height}"
        )

    result = torch.zeros((batch, new_height, new_width), device=mask.device, dtype=mask.dtype)
    up_padding = int(height * (up_factor - 1.0))
    left_padding = int(width * (left_factor - 1.0))

    target_top = max(0, up_padding)
    target_bottom = min(new_height, up_padding + height)
    target_left = max(0, left_padding)
    target_right = min(new_width, left_padding + width)
    source_top = max(0, -up_padding)
    source_bottom = min(height, new_height - up_padding)
    source_left = max(0, -left_padding)
    source_right = min(width, new_width - left_padding)

    result[:, target_top:target_bottom, target_left:target_right] = mask[
        :, source_top:source_bottom, source_left:source_right
    ]
    return result


def crop_mask_with_stitcher_geometry(
    processed_mask: torch.Tensor,
    geometry: Mapping[str, object],
    output_height: int,
    output_width: int,
    processor: object,
    downscale_algorithm: str,
    upscale_algorithm: str,
    blend_pixels: int,
) -> torch.Tensor:
    """Transform a processed source mask with the exact crop canvas geometry."""

    canvas_height, canvas_width = geometry["canvas_shape"]
    cto_x = int(geometry["canvas_to_orig_x"])
    cto_y = int(geometry["canvas_to_orig_y"])
    cto_w = int(geometry["canvas_to_orig_w"])
    cto_h = int(geometry["canvas_to_orig_h"])
    ctc_x = int(geometry["cropped_to_canvas_x"])
    ctc_y = int(geometry["cropped_to_canvas_y"])
    ctc_w = int(geometry["cropped_to_canvas_w"])
    ctc_h = int(geometry["cropped_to_canvas_h"])

    if processed_mask.shape[1:] != (cto_h, cto_w):
        raise ValueError(
            "Processed stitch mask does not match upstream canvas geometry. "
            f"Expected {(cto_h, cto_w)}, got {tuple(processed_mask.shape[1:])}"
        )

    canvas_mask = torch.zeros(
        (processed_mask.shape[0], canvas_height, canvas_width),
        device=processed_mask.device,
        dtype=processed_mask.dtype,
    )
    canvas_mask[:, cto_y : cto_y + cto_h, cto_x : cto_x + cto_w] = processed_mask
    cropped = canvas_mask[:, ctc_y : ctc_y + ctc_h, ctc_x : ctc_x + ctc_w]

    if cropped.shape[1:] != (output_height, output_width):
        algorithm = (
            upscale_algorithm
            if output_width > ctc_w or output_height > ctc_h
            else downscale_algorithm
        )
        cropped = processor.rescale_m(cropped, output_width, output_height, algorithm)

    return feather_mask(cropped, blend_pixels, processor)


def feather_mask(mask: torch.Tensor, blend_pixels: int, processor: object) -> torch.Tensor:
    """Feather a mask, including edges that touch the crop boundary."""

    if blend_pixels <= 0:
        return mask.float().clamp(0.0, 1.0)

    # Feathering cannot usefully exceed half of the available crop. Limiting the
    # working radius to that geometric maximum avoids huge temporary tensors
    # while leaving the UI control unrestricted.
    effective_blend = min(
        int(blend_pixels), max(1, min(mask.shape[1], mask.shape[2]) // 2)
    )
    padding = effective_blend
    padded = torch.zeros(
        (
            mask.shape[0],
            mask.shape[1] + padding * 2,
            mask.shape[2] + padding * 2,
        ),
        device=mask.device,
        dtype=mask.dtype,
    )
    padded[:, padding:-padding, padding:-padding] = mask
    padded = processor.blur_m(padded, effective_blend * 0.5)
    feathered = padded[
        :, padding : padding + mask.shape[1], padding : padding + mask.shape[2]
    ]

    # Gaussian blur reaches 0 outside the crop, so cropping it directly leaves
    # 0.5 on straight edges and about 0.25 at corners. Apply an inward smooth
    # edge ramp so every outermost output pixel is exactly black.
    return (feathered * full_crop_feather_mask(
        mask.shape[0],
        mask.shape[1],
        mask.shape[2],
        effective_blend,
        mask.device,
        mask.dtype,
    )).float().clamp(0.0, 1.0)


def full_crop_feather_mask(
    batch_size: int,
    height: int,
    width: int,
    blend_pixels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a full-crop rectangle that fades inward from exact-zero edges."""

    if blend_pixels <= 0:
        return torch.ones((batch_size, height, width), device=device, dtype=dtype)

    positions_y = torch.arange(height, device=device, dtype=dtype)
    positions_x = torch.arange(width, device=device, dtype=dtype)
    distance_y = torch.minimum(positions_y, (height - 1) - positions_y).view(
        height, 1
    )
    distance_x = torch.minimum(positions_x, (width - 1) - positions_x).view(
        1, width
    )
    distance_to_edge = torch.minimum(distance_y, distance_x)
    maximum_distance = float(distance_to_edge.max().item())
    effective = min(float(blend_pixels), maximum_distance)
    if effective <= 0:
        return torch.zeros((batch_size, height, width), device=device, dtype=dtype)

    normalized = (distance_to_edge / effective).clamp(0.0, 1.0)
    ramp = torch.sin(normalized * (torch.pi / 2.0))

    # Keep exactly the perimeter black. A tiny lower bound on every inner pixel
    # prevents large feathers from quantizing additional rows to zero in an
    # 8-bit ComfyUI preview.
    minimum_visible = torch.tensor(1.0 / 255.0, device=device, dtype=dtype)
    ramp = torch.where(
        distance_to_edge == 0,
        torch.zeros_like(ramp),
        torch.maximum(ramp, minimum_visible),
    )
    return ramp.view(1, height, width).expand(batch_size, -1, -1)


def stack_stitcher_masks(stitcher: Mapping[str, object]) -> torch.Tensor:
    masks = stitcher.get("cropped_mask_for_blend")
    if not isinstance(masks, list) or not masks:
        raise ValueError("Upstream stitcher does not contain cropped_mask_for_blend")
    return torch.stack([mask.squeeze(0).cpu() for mask in masks], dim=0)
