"""Gaussian image blur using only ComfyUI's existing PyTorch dependency."""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def blur_image(image, radius=8, sigma=3.0, compute_device=None):
    """Blur BHWC color channels, preserving shape, dtype, device and RGBA alpha.

    Two separable passes use replicated edges, including for 1x1 images and
    radii larger than the image. Process one batch item at a time to bound
    temporary GPU memory; the full output stays on the input device.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch tensor")
    if image.ndim != 4 or any(size <= 0 for size in image.shape):
        raise ValueError("image must have nonempty shape [B, H, W, C]")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError("image must have 1, 3 or 4 channels")
    if image.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("image must use float16, bfloat16, float32 or float64")
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 128:
        raise ValueError("radius must be an integer between 0 and 128")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and greater than 0")
    if radius == 0:
        return image

    device = image.device if compute_device is None else torch.device(compute_device)
    # CPU padding/convolution support for half types varies. Float32 also
    # avoids precision loss on CUDA; preserve float64 for double inputs.
    dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    channels = min(image.shape[-1], 3)
    result = image.clone()
    with torch.autocast(device_type=device.type, enabled=False):
        coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel /= kernel.sum()
        horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1).contiguous()
        vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1).contiguous()
        for index in range(image.shape[0]):
            work = image[index:index + 1, ..., :channels].movedim(-1, 1)
            work = work.to(device=device, dtype=dtype).contiguous()
            work = F.conv2d(
                F.pad(work, (radius, radius, 0, 0), mode="replicate"),
                horizontal, groups=channels,
            )
            work = F.conv2d(
                F.pad(work, (0, 0, radius, radius), mode="replicate"),
                vertical, groups=channels,
            )
            result[index:index + 1, ..., :channels] = work.movedim(1, -1).to(
                device=image.device, dtype=image.dtype,
            )
    return result
