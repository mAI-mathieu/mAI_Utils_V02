"""Endpoint-preserving frame selection without a ComfyUI dependency."""

from numbers import Integral


def get_ltx_frame_indices(num_frames):
    """Return evenly spaced indices for the largest 8k+1 count <= num_frames."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, Integral) or num_frames < 0:
        raise ValueError("Frame count must be a non-negative integer.")
    if num_frames == 0:
        raise ValueError("Input image batch contains no frames.")
    if num_frames < 9:
        raise ValueError(
            "At least 9 input frames are required for LTX-compatible video refinement."
        )

    target_frames = 1 + 8 * ((num_frames - 1) // 8)
    if target_frames == num_frames:
        return list(range(num_frames))

    indices = [
        round(i * (num_frames - 1) / (target_frames - 1))
        for i in range(target_frames)
    ]
    # Make the temporal endpoint contract explicit before any frame selection.
    indices[0] = 0
    indices[-1] = num_frames - 1
    return indices


def select_ltx_frames(images):
    """Select untouched frames; preserve tensor/list format and list elements.

    Lists contain one HWC or one-frame BHWC tensor per element. Compatible
    inputs are returned by identity, without allocating another image batch.
    """
    import torch

    if isinstance(images, torch.Tensor):
        if images.ndim != 4:
            raise ValueError("images must have shape [frames, height, width, channels].")
        num_frames = images.shape[0]
        indices = get_ltx_frame_indices(num_frames)
        if any(size < 1 for size in images.shape[1:]):
            raise ValueError("Image height, width, and channels must be non-empty.")
        if len(indices) == num_frames:
            return images
        tensor_indices = torch.tensor(indices, dtype=torch.long, device=images.device)
        return images.index_select(0, tensor_indices)

    if not isinstance(images, list):
        raise ValueError("images must be an IMAGE batch tensor or a list of frame tensors.")
    indices = get_ltx_frame_indices(len(images))
    for index, frame in enumerate(images):
        if not isinstance(frame, torch.Tensor) or frame.ndim not in (3, 4):
            raise ValueError(f"Image list item {index} must be an HWC or one-frame BHWC tensor.")
        if frame.ndim == 4 and frame.shape[0] != 1:
            raise ValueError(f"Image list item {index} must contain exactly one frame.")
        if any(size < 1 for size in frame.shape):
            raise ValueError(f"Image list item {index} has empty dimensions.")
        first = images[0]
        if frame.shape[-3:] != first.shape[-3:]:
            raise ValueError("All list frames must have the same height, width, and channels.")
        if frame.dtype != first.dtype or frame.device != first.device:
            raise ValueError("All list frames must have the same tensor dtype and device.")
    if len(indices) == len(images):
        return images
    return [images[index] for index in indices]
