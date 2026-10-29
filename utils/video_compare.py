"""Lossless frame previews for a comparison player; no ComfyUI dependency."""

import math
from numbers import Integral, Real
from pathlib import Path

from PIL import Image
import torch

from .frame_sequence import _get_frame_batches


def compare_batches(frames, name):
    # ComfyUI wraps even an upstream Python list when INPUT_IS_LIST is enabled.
    if isinstance(frames, (list, tuple)) and len(frames) == 1:
        if isinstance(frames[0], (list, tuple)):
            frames = frames[0]
    batches = _get_frame_batches(frames)
    for batch in batches:
        if not batch.is_floating_point() or batch.shape[-1] not in (1, 3, 4):
            raise ValueError(f"{name} must contain floating-point IMAGE frames with 1, 3 or 4 channels.")
    return batches


def compare_settings(fps, preview_max_size):
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError("fps must be a finite number greater than 0 and at most 120.")
    if isinstance(preview_max_size, bool) or not isinstance(preview_max_size, Integral) or not 64 <= preview_max_size <= 4096:
        raise ValueError("preview_max_size must be an integer between 64 and 4096.")


def preview_dimensions(width, height, max_size):
    scale = min(1.0, max_size / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def write_compare_frames(batches, folder, prefix, count, max_size, check_interrupt=None, on_frame=None):
    """Write at most one CPU frame at a time, without concatenating input batches."""
    folder = Path(folder)
    written = 0
    for batch in batches:
        for frame in batch:
            if written == count:
                return
            if check_interrupt:
                check_interrupt()
            pixels = frame.detach().to(device="cpu", dtype=torch.float32)
            pixels = torch.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
            pixels = (pixels.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            image = Image.fromarray(pixels)
            size = preview_dimensions(image.width, image.height, max_size)
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            image.save(folder / f"{prefix}_{written:06d}.png", compress_level=1)
            written += 1
            if on_frame:
                on_frame()
