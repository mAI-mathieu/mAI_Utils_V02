"""Combine optional ComfyUI IMAGE tensors without inventing missing images."""

RESIZE_METHODS = ("stretch", "preserve ratio crop", "preserve ratio pad")


def batch_connected_images(images, resize_method="stretch"):
    """Skip None and resize later batches to the first available image's size."""
    import torch

    if resize_method not in RESIZE_METHODS:
        raise ValueError(f"Unknown batch resize method: {resize_method}")

    batches = []
    for index, image in enumerate(images, start=1):
        if image is None:
            continue
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image_{index} must be an IMAGE tensor or None.")
        if image.ndim != 4 or any(size < 1 for size in image.shape):
            raise ValueError(
                f"image_{index} must have nonempty shape [batch, height, width, channels]."
            )
        if not image.is_floating_point() or image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"image_{index} must be a floating-point grayscale, RGB or RGBA image.")
        if batches:
            first = batches[0]
            if image.shape[-1] != first.shape[-1]:
                raise ValueError(
                    f"image_{index} must match the first available image's channels. "
                    "Convert images to the same color format before batching."
                )
            if image.dtype != first.dtype or image.device != first.device:
                raise ValueError(
                    f"image_{index} must match the first available image's tensor dtype and device."
                )
            if image.shape[1:3] != first.shape[1:3]:
                image = _resize_image(image, first.shape[1:3], resize_method)
        batches.append(image)

    if not batches:
        return None
    return batches[0] if len(batches) == 1 else torch.cat(batches, dim=0)


def _resize_image(image, target_size, resize_method):
    """Bilinear stretch, centered cover crop, or centered contain padding."""
    import torch
    import torch.nn.functional as functional

    target_height, target_width = target_size
    height, width = image.shape[1:3]
    resized_height, resized_width = target_size
    if resize_method == "preserve ratio crop":
        # Round the covering dimension upward so the crop always fills the target.
        if target_height * width >= target_width * height:
            resized_width = (width * target_height + height - 1) // height
        else:
            resized_height = (height * target_width + width - 1) // width
    elif resize_method == "preserve ratio pad":
        # Round the contained dimension downward so it never exceeds the target.
        if target_height * width >= target_width * height:
            resized_height = max(1, height * target_width // width)
        else:
            resized_width = max(1, width * target_height // height)

    # Interpolation operates on BCHW; ComfyUI IMAGE uses BHWC.
    samples = image.movedim(-1, 1)
    if (height, width) != (resized_height, resized_width):
        if image.dtype in (torch.float16, torch.bfloat16):
            samples = samples.float()
        samples = functional.interpolate(
            samples, size=(resized_height, resized_width),
            mode="bilinear", align_corners=False,
        )

    if resize_method == "preserve ratio crop":
        top = (resized_height - target_height) // 2
        left = (resized_width - target_width) // 2
        samples = samples[:, :, top:top + target_height, left:left + target_width]
    elif resize_method == "preserve ratio pad":
        vertical = target_height - resized_height
        horizontal = target_width - resized_width
        samples = functional.pad(samples, (
            horizontal // 2, horizontal - horizontal // 2,
            vertical // 2, vertical - vertical // 2,
        ), value=0.0)
    return samples.movedim(1, -1).to(dtype=image.dtype)
