import torch
import torch.nn.functional as torch_functional


def parse_hex_color(color):
    """Convert a #RRGGBB string to normalized RGB values."""
    if not isinstance(color, str):
        raise TypeError("color must be a string in #RRGGBB format")

    value = color.strip()
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError("color must use #RRGGBB format, for example #00FF00")

    try:
        channels = tuple(
            int(value[index : index + 2], 16) / 255.0
            for index in (1, 3, 5)
        )
    except ValueError as exc:
        raise ValueError(
            "color must use #RRGGBB format, for example #00FF00"
        ) from exc
    return channels


def _normalize_image(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[-1] < 3:
        raise ValueError(
            "image must have shape [height, width, channels] or "
            "[batch, height, width, channels] with at least 3 channels"
        )

    if not torch.is_floating_point(image):
        image = image.float()
    return image


def _normalize_mask(mask):
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")

    if mask.dim() == 2:
        return mask.unsqueeze(0)
    if mask.dim() == 3:
        return mask
    if mask.dim() == 4 and mask.shape[-1] == 1:
        return mask[..., 0]
    if mask.dim() == 4 and mask.shape[1] == 1:
        return mask[:, 0]
    raise ValueError(
        "mask must have shape [height, width], [batch, height, width], "
        "[batch, height, width, 1], or [batch, 1, height, width]"
    )


def _broadcast_batches(image, mask):
    image_batch = image.shape[0]
    mask_batch = mask.shape[0]
    if image_batch == mask_batch:
        return image, mask
    if image_batch == 1:
        return image.expand(mask_batch, -1, -1, -1), mask
    if mask_batch == 1:
        return image, mask.expand(image_batch, -1, -1)
    raise ValueError("image and mask batch sizes must match unless one batch is 1")


def _validate_outline_parameters(thickness, padding, threshold):
    if (
        not isinstance(thickness, int)
        or isinstance(thickness, bool)
        or thickness < 1
    ):
        raise ValueError("thickness must be an integer greater than or equal to 1")
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        raise ValueError("padding must be an integer greater than or equal to 0")
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or not 0 <= threshold <= 255
    ):
        raise ValueError("threshold must be an integer from 0 to 255")


def _dilate(binary_mask, radius):
    if radius == 0:
        return binary_mask

    height, width = binary_mask.shape[-2:]
    if radius >= max(height, width):
        has_mask = binary_mask.any(dim=(1, 2), keepdim=True)
        return has_mask.expand_as(binary_mask)

    values = binary_mask.unsqueeze(1).to(dtype=torch.float32)
    kernel_size = radius * 2 + 1
    values = torch_functional.max_pool2d(
        values,
        kernel_size=(1, kernel_size),
        stride=1,
        padding=(0, radius),
    )
    values = torch_functional.max_pool2d(
        values,
        kernel_size=(kernel_size, 1),
        stride=1,
        padding=(radius, 0),
    )
    return values[:, 0] > 0.5


def _erode(binary_mask, radius):
    if radius == 0:
        return binary_mask

    height, width = binary_mask.shape[-2:]
    if radius >= max(height, width):
        return torch.zeros_like(binary_mask)

    inverse = (~binary_mask).unsqueeze(1).to(dtype=torch.float32)
    kernel_size = radius * 2 + 1
    inverse = torch_functional.pad(
        inverse,
        (radius, radius, 0, 0),
        mode="constant",
        value=1.0,
    )
    inverse = torch_functional.max_pool2d(
        inverse,
        kernel_size=(1, kernel_size),
        stride=1,
    )
    inverse = torch_functional.pad(
        inverse,
        (0, 0, radius, radius),
        mode="constant",
        value=1.0,
    )
    inverse = torch_functional.max_pool2d(
        inverse,
        kernel_size=(kernel_size, 1),
        stride=1,
    )
    return inverse[:, 0] < 0.5


def create_mask_outline(mask, thickness=10, padding=0, threshold=0):
    """Create a hard contour that follows each thresholded mask shape.

    Padding expands the mask before its contour is calculated. Odd stroke widths
    place the extra pixel inside the padded mask; even widths are split evenly
    inside and outside.
    """
    _validate_outline_parameters(thickness, padding, threshold)
    normalized_mask = _normalize_mask(mask)
    binary_mask = normalized_mask > (threshold / 255.0)
    padded_mask = _dilate(binary_mask, padding)

    outside_width = thickness // 2
    inside_width = thickness - outside_width
    outside_edge = _dilate(padded_mask, outside_width)
    inside_edge = _erode(padded_mask, inside_width)
    return outside_edge & ~inside_edge


def draw_mask_outline(
    image,
    mask,
    color="#00FF00",
    thickness=10,
    padding=0,
    threshold=0,
):
    """Draw the thresholded mask contour over an image."""
    normalized_image = _normalize_image(image)
    normalized_mask = _normalize_mask(mask)
    height, width = normalized_image.shape[1:3]
    if normalized_mask.shape[1:] != (height, width):
        raise ValueError("mask dimensions must match the image dimensions")

    normalized_mask = normalized_mask.to(device=normalized_image.device)
    normalized_image, normalized_mask = _broadcast_batches(
        normalized_image, normalized_mask
    )
    outline = create_mask_outline(
        normalized_mask,
        thickness=thickness,
        padding=padding,
        threshold=threshold,
    )

    output = normalized_image.clone()
    rgb = torch.tensor(
        parse_hex_color(color),
        device=output.device,
        dtype=output.dtype,
    ).view(1, 1, 1, 3)
    output[..., :3] = torch.where(
        outline.unsqueeze(-1),
        rgb,
        output[..., :3],
    )
    return output
