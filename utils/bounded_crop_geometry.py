"""Native-resolution crop bounds that never extend outside the source image."""


def bounded_crop_box(image_width, image_height, mask_bbox, max_width, max_height, multiple_of):
    """Return (x, y, width, height), centered on the mask and shifted at edges.

    Both dimensions use the largest fitting multiple. Reject impossible mask
    coverage instead of silently dropping part of the requested inpaint area.
    An empty mask (None) selects a crop at the image center.
    """
    for name, value in (
        ("image_width", image_width), ("image_height", image_height),
        ("max_crop_width", max_width), ("max_crop_height", max_height),
        ("crop_multiple_of", multiple_of),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    width = min(image_width, max_width) // multiple_of * multiple_of
    height = min(image_height, max_height) // multiple_of * multiple_of
    if width == 0 or height == 0:
        raise ValueError(
            "The image and maximum crop dimensions must each fit at least one "
            "crop_multiple_of unit. Reduce crop_multiple_of or increase the maximum size."
        )

    if mask_bbox is None:
        return (image_width - width) // 2, (image_height - height) // 2, width, height

    if len(mask_bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in mask_bbox):
        raise ValueError("mask_bbox must contain four integer coordinates")
    left, top, right, bottom = mask_bbox
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        raise ValueError("mask_bbox must lie inside the source image")
    if right - left > width or bottom - top > height:
        raise ValueError(
            f"Mask/context bounds {right - left}x{bottom - top} do not fit the "
            f"aligned maximum crop {width}x{height}. Increase max_crop_width/height "
            "or reduce crop_multiple_of or mask expansion."
        )
    x = min(max(0, left - (width - (right - left)) // 2), image_width - width)
    y = min(max(0, top - (height - (bottom - top)) // 2), image_height - height)
    return x, y, width, height
