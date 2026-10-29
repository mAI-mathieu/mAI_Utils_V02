"""Adaptive edit-region detection, independent of the ComfyUI runtime."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .image_blur import blur_image


@dataclass(frozen=True)
class AutoEditSettings:
    """Analysis diagnostics; these are not user-supplied tuning parameters."""

    low_threshold: float
    high_threshold: float
    feather_pixels: float
    multiband_levels: int


def _local_structure_difference(original, edited):
    """Compare normalized local structure, tolerating simple contrast loss."""
    weights = original.new_tensor([0.2126, 0.7152, 0.0722])
    first = (original * weights).sum(-1, keepdim=True)
    second = (edited * weights).sum(-1, keepdim=True)
    first_mean = blur_image(first, 2)
    second_mean = blur_image(second, 2)
    first_var = (blur_image(first.square(), 2) - first_mean.square()).clamp_min(0)
    second_var = (blur_image(second.square(), 2) - second_mean.square()).clamp_min(0)
    covariance = blur_image(first * second, 2) - first_mean * second_mean
    # Stabilize near-flat patches; compare structure rather than brightness or
    # contrast, which are also affected by decoder degradation.
    stabilizer = 0.0001
    correlation = (covariance + stabilizer) / (
        (first_var * second_var).sqrt() + stabilizer
    )
    return (1 - correlation[..., 0]).clamp(0, 1), first_var[..., 0].sqrt()


def _close_regions(regions, radius, ndimage):
    # Edge padding lets an edit touch the image border without being eroded.
    padded = np.pad(regions, radius, mode="edge")
    closed = ndimage.binary_closing(padded, structure=np.ones((2 * radius + 1,) * 2))
    return closed[radius:-radius, radius:-radius]


def _local_thresholds(score, eligible, low, high):
    """Estimate spatially varying drift without allowing tiles to hide edits.

    Lower-tail tile statistics tolerate different drift in sky and foliage.
    Cap their influence relative to the whole-image estimate: an edited tile
    must not raise its own threshold enough to explain away a strong change.
    """
    height, width = score.shape
    rows, columns = max(1, math.ceil(height / 64)), max(1, math.ceil(width / 64))
    low_tiles = np.full((rows, columns), low, dtype=np.float32)
    high_tiles = np.full((rows, columns), high, dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            patch = np.s_[row * 64:min(height, (row + 1) * 64),
                          column * 64:min(width, (column + 1) * 64)]
            values = score[patch][eligible[patch]]
            if values.size < 16:
                continue
            background = values[values <= np.quantile(values, 0.7)]
            center = float(np.median(background))
            spread = float(np.median(np.abs(background - center))) * 1.4826
            low_tiles[row, column] = np.clip(center + max(0.003, 2.5 * spread), low, 2.5 * low)
            high_tiles[row, column] = np.clip(center + max(0.006, 5 * spread), high, 2.5 * high)
    def resize(values):
        return F.interpolate(torch.from_numpy(values)[None, None], size=(height, width),
                             mode="bilinear", align_corners=False)[0, 0].numpy()
    return resize(low_tiles), resize(high_tiles)


def _coherent_regions(score, eligible, low, high, ndimage):
    """Hysteresis joins weak changes to strong evidence, then fills small gaps."""
    seeds = (score > high) & eligible
    support = (score > low) & eligible
    regions = ndimage.binary_propagation(seeds, mask=support)
    radius = max(1, round(min(score.shape) / 160))
    regions = _close_regions(regions, radius, ndimage)
    regions = ndimage.binary_fill_holes(regions) & eligible
    labels, _ = ndimage.label(regions)
    areas = np.bincount(labels.ravel())
    minimum_area = max(9, round(score.size * 0.00003))
    keep = areas >= minimum_area
    keep[0] = False
    # A closing operation cannot invent a detached region without a seed.
    seeded = np.unique(labels[seeds])
    has_seed = np.zeros_like(keep)
    has_seed[seeded] = True
    has_seed[0] = False
    regions = (keep & has_seed)[labels]
    return regions


@torch.no_grad()
@torch.autocast(device_type="cpu", enabled=False)
def detect_image_edits(original_rgb, edited_rgb, allowed_mask):
    """Return a native-resolution soft mask and adaptive settings for one pair.

    Inputs are validated RGB [1,H,W,3] and an allowed MASK [1,H,W]. Analysis
    uses at most 768 pixels on the long side; connected regions and robust
    statistics run on CPU for bounded, deterministic analysis across devices.
    Most eligible pixels are assumed to be unchanged. Widespread strong
    differences without a reliable background conservatively retain the edit.
    """
    height, width = original_rgb.shape[1:3]
    native_mask = torch.zeros_like(allowed_mask)
    empty_settings = AutoEditSettings(0, 0, 0, 1)
    if not (allowed_mask > 0).any().item():
        return native_mask, empty_settings
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "Auto mode requires SciPy, normally supplied by ComfyUI. "
            "Install scipy in the ComfyUI Python environment, then restart."
        ) from exc

    scale = min(1.0, 768 / max(height, width))
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    # A shared CPU analysis also avoids unsupported half-precision reductions.
    first = original_rgb.to(device="cpu", dtype=torch.float32).movedim(-1, 1)
    second = edited_rgb.to(device="cpu", dtype=torch.float32).movedim(-1, 1)
    allowed = allowed_mask.to(device="cpu", dtype=torch.float32)[:, None]
    if size != (height, width):
        first = F.interpolate(first, size=size, mode="area")
        second = F.interpolate(second, size=size, mode="area")
        allowed = F.interpolate(allowed, size=size, mode="area")
    first, second = first.movedim(1, -1), second.movedim(1, -1)
    first_fine, second_fine = blur_image(first, 1), blur_image(second, 1)
    fine = (first_fine - second_fine).abs().mean(-1)
    broad = (blur_image(first, 3) - blur_image(second, 3)).abs().mean(-1)
    structure, texture = _local_structure_difference(first_fine, second_fine)
    color = 0.7 * fine + 0.3 * broad
    # Textured areas naturally accumulate more decoder differences. Structure
    # supplies extra evidence for moved features, rather than for lost contrast.
    score = (color * (1 + structure) / (1 + 4 * texture) + 0.01 * structure)[0].numpy()
    eligible = allowed[0, 0].numpy() > 0
    samples = score[eligible]
    background = samples[samples <= np.quantile(samples, 0.7)]
    baseline = float(np.median(background))
    deviation = float(np.median(np.abs(background - baseline))) * 1.4826
    low = max(0.006, baseline + max(0.003, 2.5 * deviation))
    high = max(0.012, baseline + max(0.006, 5 * deviation))
    local_low, local_high = _local_thresholds(score, eligible, low, high)
    regions = _coherent_regions(score, eligible, local_low, local_high, ndimage)

    # Uniform/near-uniform large changes have no unchanged population from
    # which to infer noise. Keep them instead of silently deleting the edit.
    if np.quantile(color[0].numpy()[eligible], 0.2) > 0.06:
        regions = eligible.copy()
    if not regions.any():
        return native_mask, AutoEditSettings(low, high, 0, 1)

    labels, _ = ndimage.label(regions)
    areas = np.bincount(labels.ravel())[1:]
    # Weight by area so tiny surviving regions cannot set the main seam width.
    region_size = math.sqrt(float(np.sum(areas.astype(float) ** 2) / areas.sum()))
    boundary = ndimage.binary_dilation(regions) & ~regions
    disagreement = float(np.median(color[0].numpy()[boundary])) if boundary.any() else 0
    feather = float(np.clip(1 + 0.025 * region_size + 12 * disagreement, 1, 12))
    expansion = float(np.clip(0.015 * region_size, 0.5, 4))
    # Distance transforms need an explicit outside for a completely full mask.
    if regions.all():
        soft = np.ones_like(score)
    else:
        inside = ndimage.distance_transform_edt(regions)
        outside = ndimage.distance_transform_edt(~regions)
        soft = np.clip((inside - outside + expansion + feather) / (2 * feather), 0, 1)
        soft = soft * soft * (3 - 2 * soft)
    mask = torch.from_numpy(soft.copy()).to(dtype=allowed_mask.dtype)[None, None]
    if size != (height, width):
        mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    mask = mask[:, 0].to(device=allowed_mask.device)
    # Apply the native mask last: protected holes and sub-analysis-size details
    # stay exact, and gray protection is applied only once.
    mask = (mask * allowed_mask).clamp(0, 1)
    feather_native = feather / scale
    levels = max(2, min(8, math.ceil(math.log2(max(2, 4 * feather_native)))))
    return mask, AutoEditSettings(low, high, feather_native, levels)
