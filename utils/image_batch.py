"""Combine optional ComfyUI IMAGE tensors without inventing missing images."""


def batch_connected_images(images):
    """Skip None and resize later batches to the first available image's size."""
    import torch
    import torch.nn.functional as functional

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
                # Interpolation operates on BCHW; ComfyUI IMAGE uses BHWC.
                # Compute low-precision inputs in float32 for CPU compatibility.
                samples = image.movedim(-1, 1)
                if image.dtype in (torch.float16, torch.bfloat16):
                    samples = samples.float()
                image = functional.interpolate(
                    samples,
                    size=first.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).movedim(1, -1).to(dtype=first.dtype)
        batches.append(image)

    if not batches:
        return None
    return batches[0] if len(batches) == 1 else torch.cat(batches, dim=0)
