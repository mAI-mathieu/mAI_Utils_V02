"""Batched torch resizing; no ComfyUI imports or host-side image conversions."""

from dataclasses import dataclass
import math
from numbers import Integral

import torch
import torch.nn.functional as F


NATIVE_METHODS = ("nearest", "nearest-exact", "bilinear", "bicubic", "area")
CUSTOM_METHODS = ("lanczos2", "lanczos3", "lanczos4", "mitchell", "catmull_rom")
RESIZE_METHODS = ("auto",) + NATIVE_METHODS + CUSTOM_METHODS
RESIZE_MODES = ("exact", "keep_aspect_fit", "keep_aspect_fill")
# Custom passes hold only one sampled plane per tap. Bound their batch working
# set conservatively, independently of free VRAM, without probing for OOM.
CUSTOM_WORKING_BYTES = 512 * 1024 * 1024


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(value)


def apply_multiple_of(dimension, multiple_of=1):
    """Nearest multiple, with ties rounded up and a minimum of one multiple."""
    dimension = _positive_integer(dimension, "Dimension")
    multiple_of = _positive_integer(multiple_of, "multiple_of")
    return max(multiple_of, ((dimension + multiple_of // 2) // multiple_of) * multiple_of)


@dataclass(frozen=True)
class ResizeGeometry:
    width: int
    height: int
    resize_width: int
    resize_height: int


def calculate_target_size(source_width, source_height, width, height,
                          resize_mode="exact", multiple_of=1):
    """Round the final canvas first, then derive a centered fit/fill geometry."""
    source_width = _positive_integer(source_width, "Source width")
    source_height = _positive_integer(source_height, "Source height")
    width = apply_multiple_of(width, multiple_of)
    height = apply_multiple_of(height, multiple_of)
    if resize_mode not in RESIZE_MODES:
        raise ValueError(f"Unknown resize mode: {resize_mode}")
    if resize_mode == "exact":
        return ResizeGeometry(width, height, width, height)
    width_limits = width * source_height <= height * source_width
    if resize_mode == "keep_aspect_fit":
        if width_limits:
            rh = max(1, (2 * source_height * width + source_width) // (2 * source_width))
            return ResizeGeometry(width, height, width, min(height, rh))
        rw = max(1, (2 * source_width * height + source_height) // (2 * source_height))
        return ResizeGeometry(width, height, min(width, rw), height)
    if width_limits:
        rw = (source_width * height + source_height - 1) // source_height
        return ResizeGeometry(width, height, max(width, rw), height)
    rh = (source_height * width + source_width - 1) // source_width
    return ResizeGeometry(width, height, width, max(height, rh))


def select_auto_resize_method(source_width, source_height, target_width, target_height):
    if (source_width, source_height) == (target_width, target_height):
        return "identity"
    scales = (target_width / source_width, target_height / source_height)
    if min(scales) <= 0.65:
        return "lanczos3"
    if min(scales) < 1.0:
        return "lanczos2"
    return "bicubic" if max(scales) <= 1.5 else "catmull_rom"


def lanczos_kernel(distance, radius):
    return torch.where(distance.abs() < radius,
                       torch.sinc(distance) * torch.sinc(distance / radius), 0.0)


def bc_cubic_kernel(distance, b, c):
    """Mitchell-Netravali BC cubic, supported on [-2, 2]."""
    x = distance.abs()
    inner = ((12 - 9*b - 6*c)*x**3 + (-18 + 12*b + 6*c)*x**2 + 6 - 2*b) / 6
    outer = ((-b - 6*c)*x**3 + (6*b + 30*c)*x**2
             + (-12*b - 48*c)*x + 8*b + 24*c) / 6
    return torch.where(x < 1, inner, torch.where(x < 2, outer, 0.0))


def build_sampling_weights(source_size, target_size, method, antialias, device,
                           dtype=torch.float32):
    """Shared half-pixel sampling table [tap, output], entirely on device.

    Shrinking widens support by source/target when antialiasing is enabled.
    Distances are evaluated BEFORE clamping indices: out-of-bounds taps replicate
    the edge, then normalized weights preserve DC/constant colors.
    """
    if method not in CUSTOM_METHODS:
        raise ValueError(f"Unknown custom resize method: {method}")
    radius = int(method[-1]) if method.startswith("lanczos") else 2
    stretch = max(1.0, source_size / target_size) if antialias else 1.0
    support = radius * stretch
    taps = math.ceil(2 * support) + 1
    centers = (torch.arange(target_size, device=device, dtype=dtype) + 0.5)
    centers = centers * (source_size / target_size) - 0.5
    starts = torch.floor(centers - support).to(torch.int64)
    indices = starts[None, :] + torch.arange(taps, device=device)[:, None]
    distance = (centers[None, :] - indices.to(dtype)) / stretch
    if method.startswith("lanczos"):
        weights = lanczos_kernel(distance, radius)
    else:
        b, c = (1/3, 1/3) if method == "mitchell" else (0.0, 0.5)
        weights = bc_cubic_kernel(distance, b, c)
    weights = weights / weights.sum(dim=0, keepdim=True)
    return indices.clamp_(0, source_size - 1), weights


def _filter_axis(x, axis, table):
    if table is None:
        return x
    indices, weights = table
    shape = list(x.shape)
    shape[axis] = indices.shape[1]
    result = x.new_zeros(shape)
    broadcast = [1] * x.ndim
    broadcast[axis] = indices.shape[1]
    # A small loop over taps, never frames/pixels. index_select plus addcmul
    # avoids materializing [B,C,H,W,taps] and keeps accumulation in FP32/FP64.
    for tap in range(indices.shape[0]):
        sampled = x.index_select(axis, indices[tap])
        result.addcmul_(sampled, weights[tap].view(broadcast))
        del sampled
    return result


def resize_separable(x, tables, horizontal_first=True):
    horizontal, vertical = tables
    axes = ((3, horizontal), (2, vertical)) if horizontal_first else ((2, vertical), (3, horizontal))
    for axis, table in axes:
        x = _filter_axis(x, axis, table)
    return x


def resize_native(x, size, method, antialias=True):
    kwargs = {"size": size, "mode": method}
    if method in ("bilinear", "bicubic"):
        kwargs.update(align_corners=False, antialias=antialias)
    # Antialiased interpolate does not support reduced precision on all torch
    # backends. CPU linear/cubic/area also use a portable FP32 working dtype.
    reduced = x.dtype in (torch.float16, torch.bfloat16)
    promote = reduced and ((method in ("bilinear", "bicubic") and antialias)
                          or (x.device.type == "cpu" and method not in ("nearest", "nearest-exact")))
    return F.interpolate(x.float() if promote else x, **kwargs)


def _finish_geometry(x, geometry, resize_mode):
    if resize_mode == "keep_aspect_fit":
        dw = geometry.width - x.shape[3]
        dh = geometry.height - x.shape[2]
        if dw or dh:
            return F.pad(x, (dw//2, dw - dw//2, dh//2, dh - dh//2), value=0.0)
    elif resize_mode == "keep_aspect_fill":
        left = (x.shape[3] - geometry.width) // 2
        top = (x.shape[2] - geometry.height) // 2
        return x[:, :, top:top + geometry.height, left:left + geometry.width]
    return x


def resolve_output_dtype(input_dtype, precision):
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision == "auto":
        return input_dtype
    if precision not in dtypes:
        raise ValueError(f"Unknown precision: {precision}")
    return dtypes[precision]


def choose_chunk_size(batch, channels, source_size, resize_size, output_size,
                      chunk_size, custom, element_size=4):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 0:
        raise ValueError("chunk_size must be an integer >= 0.")
    if chunk_size:
        return min(batch, int(chunk_size))
    if not custom:
        return batch
    sh, sw = source_size
    rh, rw = resize_size
    oh, ow = output_size
    intermediate = min(sh * rw, rh * sw)
    # Source cast + inter-pass storage + sampled/output planes + geometry/cast.
    pixels = sh*sw + intermediate + 3*max(intermediate, rh*rw) + oh*ow
    per_frame = channels * pixels * element_size
    return min(batch, max(1, CUSTOM_WORKING_BYTES // per_frame))


@torch.inference_mode()
def resize_image_batch(image, width, height, resize_mode="exact", method="auto",
                       antialias=True, multiple_of=1, compute_device=None,
                       precision="auto", chunk_size=0):
    """Return BHWC IMAGE, width, height, actual method; preserve frame ordering."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or any(d < 1 for d in image.shape):
        raise ValueError("image must be a non-empty torch tensor [batch, height, width, channels].")
    if image.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("image must use float32, float16, bfloat16 or float64 pixels.")
    if method not in RESIZE_METHODS:
        raise ValueError(f"Unknown resize method: {method}")
    if not isinstance(antialias, bool):
        raise ValueError("antialias must be a boolean.")
    geometry = calculate_target_size(image.shape[2], image.shape[1], width, height, resize_mode, multiple_of)
    device = image.device if compute_device is None else torch.device(compute_device)
    dtype = resolve_output_dtype(image.dtype, precision)
    same_size = (image.shape[2], image.shape[1]) == (geometry.resize_width, geometry.resize_height)
    method_used = "identity" if same_size else method
    if method_used == "auto":
        method_used = select_auto_resize_method(image.shape[2], image.shape[1], geometry.resize_width, geometry.resize_height)
    custom = method_used in CUSTOM_METHODS
    work_dtype = (torch.float64 if dtype == torch.float64 else torch.float32) if custom else dtype
    size = (geometry.resize_height, geometry.resize_width)
    chunks = choose_chunk_size(image.shape[0], image.shape[3], image.shape[1:3], size,
                               (geometry.height, geometry.width), chunk_size, custom,
                               8 if work_dtype == torch.float64 else 4)
    if same_size and image.shape[1:3] == (geometry.height, geometry.width):
        # .to is an identity when both dtype and device already match.
        return image.to(device=device, dtype=dtype), geometry.width, geometry.height, "identity"
    tables = (None, None)
    horizontal_first = image.shape[1] * geometry.resize_width <= geometry.resize_height * image.shape[2]
    if custom:
        tables = tuple(
            None if src == dst else build_sampling_weights(src, dst, method_used, antialias, device, work_dtype)
            for src, dst in ((image.shape[2], geometry.resize_width), (image.shape[1], geometry.resize_height))
        )

    def process(batch):
        x = batch.to(device=device, dtype=work_dtype).permute(0, 3, 1, 2)
        if custom:
            x = resize_separable(x, tables, horizontal_first)
        elif method_used != "identity":
            x = resize_native(x, size, method_used, antialias)
        if custom or method_used == "bicubic":
            x.clamp_(0.0, 1.0)  # Only the final reconstruction is range limited.
        x = _finish_geometry(x, geometry, resize_mode)
        return x.permute(0, 2, 3, 1).to(dtype=dtype)

    if chunks >= image.shape[0]:
        result = process(image)
    else:
        # Preallocate once, avoiding torch.cat's extra full-batch output copy.
        result = torch.empty((image.shape[0], geometry.height, geometry.width, image.shape[3]),
                             dtype=dtype, device=device)
        for start in range(0, image.shape[0], chunks):
            end = min(start + chunks, image.shape[0])
            result[start:end] = process(image[start:end])
    return result, geometry.width, geometry.height, method_used
