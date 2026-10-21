"""Bounded native crops with the installed CropAndStitch v3 stitcher format."""

import torch

from .bounded_crop_geometry import bounded_crop_layout
from .separate_stitch_mask import (
    broadcast_inputs, feather_mask, full_crop_feather_mask, normalize_mask,
)


def bounded_inpaint_crop(inputs, processor, device, max_width, max_height, multiple_of,
                         stitch_mask_mode, stitch_expand_pixels, stitch_blend_pixels):
    """Keep masks complete and downscale oversize source regions to output limits."""
    source_image = inputs["image"]
    if source_image.ndim != 4:
        raise ValueError("image must have shape [B,H,W,C]")

    def source_sized_mask(value, name):
        if value is None:
            return None
        value = normalize_mask(value, name)
        # LoadImage may emit a 64x64 empty placeholder for a larger RGB image.
        # Match upstream's fallback while still rejecting nonempty misalignment.
        if value.shape[1:] != source_image.shape[1:3] and not torch.count_nonzero(value):
            return torch.zeros(
                (value.shape[0], *source_image.shape[1:3]), device=value.device, dtype=value.dtype,
            )
        return value

    source_mask = source_sized_mask(inputs.get("mask"), "mask")
    source_context = source_sized_mask(inputs.get("optional_context_mask"), "optional_context_mask")
    if source_mask is None:
        source_mask = torch.zeros_like(source_image[..., 0])
    image, render_mask, context_mask, stitch_source = broadcast_inputs(
        source_image, source_mask, source_context, source_mask,
    )
    image_height, image_width = image.shape[1:3]
    stitcher = {
        "downscale_algorithm": inputs["downscale_algorithm"],
        "upscale_algorithm": inputs["upscale_algorithm"],
        "blend_pixels": stitch_blend_pixels,
        "device_mode": inputs["device_mode"],
        **{name: [] for name in (
            "canvas_to_orig_x", "canvas_to_orig_y", "canvas_to_orig_w", "canvas_to_orig_h",
            "canvas_image", "cropped_to_canvas_x", "cropped_to_canvas_y",
            "cropped_to_canvas_w", "cropped_to_canvas_h", "cropped_mask_for_blend",
        )},
    }
    processed_masks, stitch_masks, bboxes = [], [], []
    for index in range(image.shape[0]):
        mask = render_mask[index:index + 1].to(device)
        context = context_mask[index:index + 1].to(device)
        # Match upstream's order; the render and stitch masks remain independent.
        if inputs["mask_fill_holes"]:
            mask = processor.fillholes_iterative_hipass_fill_m(mask)
        if inputs["mask_expand_pixels"] > 0:
            mask = processor.expand_m(mask, inputs["mask_expand_pixels"])
        if inputs["mask_invert"]:
            mask = processor.invert_m(mask)
        if inputs["mask_blend_pixels"] > 0:
            mask = processor.expand_m(mask, inputs["mask_blend_pixels"])
            mask = processor.blur_m(mask, inputs["mask_blend_pixels"] * 0.5)
        if inputs["mask_hipass_filter"] >= 0.01:
            mask = processor.hipassfilter_m(mask, inputs["mask_hipass_filter"])
            context = processor.hipassfilter_m(context, inputs["mask_hipass_filter"])

        area = (mask > 0) | (context > 0)
        stitch_mask = None
        if stitch_mask_mode == "extended mask":
            stitch_mask = stitch_source[index:index + 1].to(device)
            if stitch_expand_pixels > 0:
                stitch_mask = processor.expand_m(stitch_mask, stitch_expand_pixels)
            area |= stitch_mask > 0

        # Reduce each axis before transferring coordinates from GPU to CPU.
        rows = torch.where(area.any(dim=2)[0])[0]
        columns = torch.where(area.any(dim=1)[0])[0]
        bbox = None
        if rows.numel():
            bbox = (int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)
        processed_masks.append(mask)
        stitch_masks.append(stitch_mask)
        bboxes.append(bbox)

    boxes, output_width, output_height = bounded_crop_layout(
        image_width, image_height, bboxes, max_width, max_height, multiple_of,
    )
    crops, masks, blend_masks = [], [], []
    for index, (x, y, width, height) in enumerate(boxes):
        cropped_image = image[index:index + 1, y:y + height, x:x + width].to(device)
        cropped_mask = processed_masks[index][:, y:y + height, x:x + width]
        stitch_mask = stitch_masks[index]
        if stitch_mask is not None:
            stitch_mask = stitch_mask[:, y:y + height, x:x + width]
        if (width, height) != (output_width, output_height):
            cropped_image = processor.rescale_i(
                cropped_image, output_width, output_height, inputs["downscale_algorithm"],
            )
            cropped_mask = processor.rescale_m(
                cropped_mask, output_width, output_height, inputs["downscale_algorithm"],
            )
            if stitch_mask is not None:
                stitch_mask = processor.rescale_m(
                    stitch_mask, output_width, output_height, inputs["downscale_algorithm"],
                )
        if stitch_mask is None:
            blend_mask = full_crop_feather_mask(
                1, output_height, output_width, stitch_blend_pixels, device, torch.float32,
            )
        else:
            blend_mask = feather_mask(stitch_mask, stitch_blend_pixels, processor)
        blend_mask = blend_mask.cpu()
        geometry = {
            "canvas_to_orig_x": 0, "canvas_to_orig_y": 0,
            "canvas_to_orig_w": image_width, "canvas_to_orig_h": image_height,
            "canvas_image": image[index:index + 1].cpu().clone(),
            "cropped_to_canvas_x": x, "cropped_to_canvas_y": y,
            "cropped_to_canvas_w": width, "cropped_to_canvas_h": height,
            "cropped_mask_for_blend": blend_mask,
        }
        for name, value in geometry.items():
            stitcher[name].append(value)
        crops.append(cropped_image.cpu().clone())
        masks.append(cropped_mask.cpu().clone())
        blend_masks.append(blend_mask)
    return stitcher, torch.cat(crops), torch.cat(masks), torch.cat(blend_masks)
