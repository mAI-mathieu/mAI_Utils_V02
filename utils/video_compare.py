"""GPU-first frame previews for a comparison player; no ComfyUI dependency."""

import math
from numbers import Integral, Real
from pathlib import Path
import sys

from PIL import Image
import torch

from .frame_sequence import _get_frame_batches
from .resize_kernels import build_sampling_weights, resize_separable


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


def resolve_compare_device(image_device):
    """Prefer ComfyUI's CUDA device, even when a loader supplies CPU frames."""
    management = sys.modules.get("comfy.model_management")
    get_device = getattr(management, "get_torch_device", None)
    if get_device:
        preferred = torch.device(get_device())
        # Respect --cpu without even querying CUDA.
        if preferred.type != "cuda":
            return torch.device("cpu")
    else:
        source = torch.device(image_device)
        preferred = source if source.type == "cuda" else torch.device("cuda")
    return preferred if torch.cuda.is_available() else torch.device("cpu")


@torch.inference_mode()
def prepare_compare_pixels(frame, device, tables=(None, None)):
    """Return preview uint8 pixels on the compute device; never mutate inputs."""
    pixels = frame.detach().to(device=device, dtype=torch.float32)
    pixels = torch.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
    if any(table is not None for table in tables):
        pixels = pixels.permute(2, 0, 1).unsqueeze(0)
        # Premultiply alpha while filtering, as Pillow's RGBA resize does.
        if pixels.shape[1] == 4:
            pixels = torch.cat((pixels[:, :3] * pixels[:, 3:], pixels[:, 3:]), dim=1)
        pixels = resize_separable(pixels, tables)
        if pixels.shape[1] == 4:
            alpha = pixels[:, 3:].clamp(0, 1)
            rgb = torch.where(alpha > 1e-6, pixels[:, :3] / alpha.clamp_min(1e-6), 0)
            pixels = torch.cat((rgb, alpha), dim=1)
        pixels = pixels.squeeze(0).permute(1, 2, 0)
    return pixels.clamp(0, 1).mul_(255).round_().to(torch.uint8)


def write_compare_frames(batches, folder, prefix, count, max_size, check_interrupt=None,
                         on_frame=None, compute_device=None):
    """Resize/quantize on GPU one frame at a time; PNG compression stays on CPU."""
    folder = Path(folder)
    device = resolve_compare_device(batches[0].device) if compute_device is None else torch.device(compute_device)
    height, width = batches[0].shape[1:3]
    size = preview_dimensions(width, height, max_size)
    # Reuse Lanczos tables for every frame; avoid per-frame CUDA memory cleanup
    # and never copy the entire input video to GPU or concatenate input batches.
    tables = tuple(
        None if source == target else build_sampling_weights(source, target, "lanczos3", True, device)
        for source, target in zip((width, height), size)
    )
    written = 0
    for batch in batches:
        for frame in batch:
            if written == count:
                return
            if check_interrupt:
                check_interrupt()
            # Only the downscaled, packed uint8 preview crosses back to CPU.
            pixels = prepare_compare_pixels(frame, device, tables).cpu().numpy()
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            image = Image.fromarray(pixels)
            image.save(folder / f"{prefix}_{written:06d}.png", compress_level=1)
            written += 1
            if on_frame:
                on_frame()
