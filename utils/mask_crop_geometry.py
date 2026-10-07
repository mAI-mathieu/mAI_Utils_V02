"""Integer source-space geometry, independent of torch and ComfyUI."""

from math import gcd


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def valid_crop_bounds(context):
    """Return the half-open intersection of a crop with the original canvas."""
    return (
        max(0, context["crop_x"]),
        max(0, context["crop_y"]),
        min(context["original_width"], context["crop_x"] + context["source_crop_width"]),
        min(context["original_height"], context["crop_y"] + context["source_crop_height"]),
    )


def calculate_crop_geometry(original_width, original_height, mask_bbox,
                            output_width, output_height, mask_padding=0,
                            edge_mode="shift", allow_upscale=False):
    """Plan a native crop, enlarging only when the padded bbox cannot fit.

    Bounds use (left, top, right, bottom), with exclusive right/bottom.
    Extra odd context pixels go to the right/bottom. Resized rectangles use
    integer multiples of the reduced target aspect ratio: X/Y scales agree
    exactly, including unusual rectangular targets.
    """
    for name, value in (("original_width", original_width),
                        ("original_height", original_height),
                        ("output_width", output_width), ("output_height", output_height)):
        _positive_integer(value, name)
    _positive_integer(mask_padding, "mask_padding", minimum=0)
    if edge_mode not in ("shift", "pad"):
        raise ValueError(f"Unknown edge_mode: {edge_mode}")
    if not isinstance(allow_upscale, bool):
        raise ValueError("allow_upscale must be a boolean")

    padded_bbox = None
    if mask_bbox is not None:
        if len(mask_bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, int)
                                      for v in mask_bbox):
            raise ValueError("mask_bbox must contain four integer coordinates")
        left, top, right, bottom = mask_bbox
        if not (0 <= left < right <= original_width and
                0 <= top < bottom <= original_height):
            raise ValueError("mask_bbox must lie inside the original image")
        padded_bbox = (max(0, left - mask_padding), max(0, top - mask_padding),
                       min(original_width, right + mask_padding),
                       min(original_height, bottom + mask_padding))
        left, top, right, bottom = padded_bbox
        bbox_width, bbox_height = right - left, bottom - top
    else:
        bbox_width = bbox_height = 0

    divisor = gcd(output_width, output_height)
    unit_width, unit_height = output_width // divisor, output_height // divisor
    required_units = max((bbox_width + unit_width - 1) // unit_width,
                         (bbox_height + unit_height - 1) // unit_height, 1)
    units = divisor
    if bbox_width > output_width or bbox_height > output_height:
        units = required_units
    elif allow_upscale and (original_width < output_width or original_height < output_height):
        fitting_units = min(original_width // unit_width, original_height // unit_height)
        # Upscale only a rectangle that still contains the complete padded mask.
        if required_units <= fitting_units < divisor:
            units = fitting_units

    source_width, source_height = unit_width * units, unit_height * units
    if padded_bbox is None:
        crop_x = (original_width - source_width) // 2
        crop_y = (original_height - source_height) // 2
    else:
        crop_x = left - (source_width - bbox_width) // 2
        crop_y = top - (source_height - bbox_height) // 2

    if edge_mode == "shift":
        # For oversize rectangles this interval retains the entire image;
        # otherwise it keeps the rectangle entirely within the image.
        crop_x = min(max(crop_x, min(0, original_width - source_width)),
                     max(0, original_width - source_width))
        crop_y = min(max(crop_y, min(0, original_height - source_height)),
                     max(0, original_height - source_height))

    pad_left, pad_top = max(0, -crop_x), max(0, -crop_y)
    pad_right = max(0, crop_x + source_width - original_width)
    pad_bottom = max(0, crop_y + source_height - original_height)
    return {
        "original_width": original_width, "original_height": original_height,
        "crop_x": crop_x, "crop_y": crop_y,
        "source_crop_width": source_width, "source_crop_height": source_height,
        "output_width": output_width, "output_height": output_height,
        "scale": divisor / units,
        "pad_left": pad_left, "pad_right": pad_right,
        "pad_top": pad_top, "pad_bottom": pad_bottom,
        "mask_bbox": mask_bbox, "padded_mask_bbox": padded_bbox,
        "was_resized": units != divisor,
        "was_padded": any((pad_left, pad_right, pad_top, pad_bottom)),
    }
