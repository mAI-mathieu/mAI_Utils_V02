"""Native-resolution crop bounds that never extend outside the source image."""


def bounded_crop_box(image_width, image_height, mask_bbox, max_width, max_height, multiple_of):
    """Return (x, y, width, height), centered on the mask and shifted at edges.

    Start with the largest fitting multiples. If the mask is larger, retain its
    full source bounds; bounded_crop_layout sizes the downscaled output.
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
    width = max(width, right - left)
    height = max(height, bottom - top)
    x = min(max(0, left - (width - (right - left)) // 2), image_width - width)
    y = min(max(0, top - (height - (bottom - top)) // 2), image_height - height)
    return x, y, width, height


def bounded_crop_layout(image_width, image_height, mask_bboxes, max_width, max_height, multiple_of):
    """Plan shared batch dimensions and independently positioned source crops.

    Oversize masks grow the source rectangle inside the image, then downscale to
    the output limits. Add context to match the output aspect ratio when space
    permits; otherwise round scaled dimensions down to the requested multiple.
    """
    _, _, output_width, output_height = bounded_crop_box(
        image_width, image_height, None, max_width, max_height, multiple_of,
    )
    if not mask_bboxes:
        raise ValueError("At least one image/mask batch item is required")
    source_boxes = [bounded_crop_box(
        image_width, image_height, bbox, max_width, max_height, multiple_of,
    ) for bbox in mask_bboxes]
    source_width = max(box[2] for box in source_boxes)
    source_height = max(box[3] for box in source_boxes)
    if source_width > output_width or source_height > output_height:
        # Grow context rather than stretch the mask into the target aspect.
        if source_width * output_height < source_height * output_width:
            source_width = min(
                image_width, (source_height * output_width + output_height - 1) // output_height,
            )
        else:
            source_height = min(
                image_height, (source_width * output_height + output_width - 1) // output_width,
            )
        # Integer arithmetic avoids one-pixel float errors at aligned limits.
        if source_width * output_height >= source_height * output_width:
            numerator, denominator = output_width, source_width
        else:
            numerator, denominator = output_height, source_height
        output_width = max(multiple_of, source_width * numerator // denominator // multiple_of * multiple_of)
        output_height = max(multiple_of, source_height * numerator // denominator // multiple_of * multiple_of)
    boxes = [bounded_crop_box(
        image_width, image_height, bbox, source_width, source_height, 1,
    ) for bbox in mask_bboxes]
    return boxes, output_width, output_height
