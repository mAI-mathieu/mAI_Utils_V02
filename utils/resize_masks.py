"""Mask validation and transforms shared with tensor image resizing."""

import torch

from .resize_kernels import resize_image_batch


def prepare_resize_mask(mask, image_shape):
    """Require source-size masks; broadcast one mask across an image batch."""
    if mask is None:
        return None
    if len(image_shape) != 4 or any(size < 1 for size in image_shape):
        raise ValueError("image must be a non-empty torch tensor [batch, height, width, channels].")
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3) or any(size < 1 for size in mask.shape):
        raise ValueError("mask must be a non-empty torch tensor [batch, height, width] or [height, width].")
    if mask.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("mask must use float32, float16, bfloat16 or float64 values.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[1:3] != image_shape[1:3]:
        raise ValueError("mask width and height must match the source image before resizing.")
    if mask.shape[0] not in (1, image_shape[0]):
        raise ValueError("mask batch must contain one mask or match the image batch size.")
    return mask.expand(image_shape[0], -1, -1)


@torch.inference_mode()
def resize_mask_batch(mask, output_image, width, height, resize_mode="exact", method="auto",
                      antialias=True, multiple_of=1, precision="auto", chunk_size=0):
    """Transform a prepared BHW mask like its image, or return an empty mask."""
    if mask is None:
        return output_image.new_zeros(output_image.shape[:3])
    result, _, _, _ = resize_image_batch(
        mask.unsqueeze(-1), width, height, resize_mode, method, antialias, multiple_of,
        output_image.device, precision, chunk_size,
    )
    return result.squeeze(-1)
