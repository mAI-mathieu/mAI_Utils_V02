"""Combine optional ComfyUI IMAGE tensors without inventing missing images."""


def batch_connected_images(images):
    """Skip None, preserve input order, and return None when all inputs are missing."""
    import torch

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
            if image.shape[1:] != first.shape[1:]:
                raise ValueError(
                    f"image_{index} must match the first available image's height, width and channels. "
                    "Resize or convert images before batching."
                )
            if image.dtype != first.dtype or image.device != first.device:
                raise ValueError(
                    f"image_{index} must match the first available image's tensor dtype and device."
                )
        batches.append(image)

    if not batches:
        return None
    return batches[0] if len(batches) == 1 else torch.cat(batches, dim=0)
