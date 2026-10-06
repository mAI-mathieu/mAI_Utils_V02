"""Tensor-only loop search. No ComfyUI, codecs, models, or import-time CUDA calls.

See docs/auto_seamless_loop.md for the objective and approximation limits.
"""

from dataclasses import asdict, dataclass
from itertools import groupby
import math
from numbers import Integral
import time

import torch
import torch.nn.functional as F


QUALITY = {"fast": (64, 128, 4), "balanced": (96, 256, 8), "high": (128, 384, 16)}
METRICS = ("appearance", "exposure", "motion", "smoothness", "ghosting", "duration", "fade", "activity_loss", "contrast")


@dataclass(frozen=True, order=True)
class LoopCandidate:
    trim_start: int = 0
    trim_end: int = 0
    overlap: int = 0


@dataclass(frozen=True)
class LoopOptions:
    fps: float = 24.0
    device: str = "auto"
    quality: str = "balanced"
    max_trim_start: int = 8
    max_trim_end: int = 8
    max_fade: int = 12
    min_retained_percent: float = 70.0
    blend_space: str = "srgb"
    appearance_weight: float = 1.0
    exposure_weight: float = 0.5
    motion_weight: float = 0.6
    smoothness_weight: float = 0.4
    ghosting_weight: float = 1.0
    duration_weight: float = 0.25
    fade_weight: float = 0.12
    satisfactory_score: float = 0.025
    tie_tolerance: float = 0.002
    manual_trim_start: int = -1
    manual_trim_end: int = -1
    manual_overlap: int = -1
    candidate_chunk: int = 16
    render_chunk: int = 8
    max_candidates: int = 50000


def validate_options(options):
    for name in ("fps", "min_retained_percent", "satisfactory_score", "tie_tolerance"):
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if options.fps <= 0:
        raise ValueError("fps must be positive.")
    if not 0 < options.min_retained_percent <= 100:
        raise ValueError("min_retained_percent must be in (0, 100].")
    for name in ("max_trim_start", "max_trim_end", "max_fade", "candidate_chunk", "render_chunk", "max_candidates",
                 "manual_trim_start", "manual_trim_end", "manual_overlap"):
        value = getattr(options, name)
        minimum = -1 if name.startswith("manual") else (1 if name.endswith("chunk") or name == "max_candidates" else 0)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name in ("appearance", "exposure", "motion", "smoothness", "ghosting", "duration", "fade"):
        value = getattr(options, name + "_weight")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name}_weight must be finite and nonnegative.")
    if options.satisfactory_score < 0 or options.tie_tolerance < 0:
        raise ValueError("Score thresholds must be nonnegative.")
    if options.device not in ("auto", "cuda", "cpu") or options.quality not in QUALITY:
        raise ValueError("Unknown device or quality.")
    if options.blend_space not in ("srgb", "linear"):
        raise ValueError("blend_space must be srgb or linear.")


def _validate_image_structure(images):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or min(images.shape) < 1:
        raise ValueError("images must be a nonempty [N,H,W,C] tensor.")
    if not images.is_floating_point():
        raise ValueError("images must have a floating-point dtype with pixels in [0,1].")


def validate_images(images, chunk=8, check_interrupt=None, device=None):
    _validate_image_structure(images)
    device = torch.device(device or images.device)
    valid = torch.ones((), dtype=torch.bool, device=device)
    for offset in range(0, len(images), chunk):
        if check_interrupt:
            check_interrupt()
        part = images[offset:offset + chunk].to(device)
        low, high = torch.aminmax(part)
        # Reductions avoid full-size boolean masks; NaN and Inf fail these bounds.
        valid &= (low >= 0) & (high <= 1)
    if not bool(valid):
        raise ValueError("images contain NaN/Inf or pixels outside [0,1].")


def enumerate_candidates(frame_count, options):
    """Exhaustive bounded integer grid; constraints apply to final cycle length."""
    validate_options(options)
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count < 1:
        raise ValueError("frame_count must be positive.")
    minimum = math.ceil(frame_count * options.min_retained_percent / 100)
    bounds = (min(options.max_trim_start, frame_count - minimum),
              min(options.max_trim_end, frame_count - minimum),
              min(options.max_fade, (frame_count - 1) // 2, frame_count - minimum))
    manual = (options.manual_trim_start, options.manual_trim_end, options.manual_overlap)
    for value, bound in zip(manual, bounds):
        if value > bound:
            raise ValueError("Manual override exceeds search bounds or retained-duration constraints.")
    ranges = [range(bound + 1) if value == -1 else (value,) for value, bound in zip(manual, bounds)]
    candidates = []
    for a in ranges[0]:
        for b in ranges[1]:
            length = frame_count - a - b
            for k in ranges[2]:
                if length - k >= minimum and (k == 0 or length >= 2 * k + 1):
                    candidates.append(LoopCandidate(a, b, k))
                    if len(candidates) > options.max_candidates:
                        raise ValueError(f"Search exceeds {options.max_candidates} candidates; reduce trim/fade limits.")
    if not candidates:
        raise ValueError("Impossible search constraints: no valid repeatable cycle remains.")
    return candidates


def blend_weights(k, device="cpu"):
    """Inclusive linear ramp: K>=2 has exact tail/head endpoints; K=1 uses 1/2."""
    if k < 1:
        raise ValueError("Overlap must be positive for blend weights.")
    return torch.full((1,), 0.5, device=device) if k == 1 else torch.linspace(0, 1, k, device=device)


def blend_frames(tail, head, alpha, space="srgb"):
    tail, head = tail.float(), head.float()
    if space == "srgb":
        return torch.lerp(tail, head, alpha)
    if space != "linear":
        raise ValueError("Unknown blend space.")
    # sRGB transfer applies to color channels only; alpha (RGBA) stays linear.
    channels = min(3, tail.shape[-1])
    t, h = tail[..., :channels], head[..., :channels]
    t = torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055).pow(2.4))
    h = torch.where(h <= 0.04045, h / 12.92, ((h + 0.055) / 1.055).pow(2.4))
    value = torch.lerp(t, h, alpha)
    value = torch.where(value <= 0.0031308, value * 12.92, 1.055 * value.clamp_min(0).pow(1 / 2.4) - 0.055)
    # Preserve exact endpoints rather than introducing transfer-function roundoff.
    value = torch.where(alpha == 0, tail[..., :channels], torch.where(alpha == 1, head[..., :channels], value))
    if tail.shape[-1] > channels:
        value = torch.cat((value, torch.lerp(tail[..., channels:], head[..., channels:], alpha)), dim=-1)
    return value.clamp(0, 1)


@torch.inference_mode()
def render_cycle(images, candidate, blend_space="srgb", device=None, output_device=None, chunk=8,
                 check_interrupt=None, on_progress=None):
    """Return T[K:L-K] followed by (1-w)*T[L-K:L]+w*T[0:K]."""
    if chunk < 1:
        raise ValueError("Render chunk must be positive.")
    a, b, k = candidate.trim_start, candidate.trim_end, candidate.overlap
    length = len(images) - a - b
    if min(a, b, k) < 0 or length < 1 or (k and length < 2 * k + 1):
        raise ValueError("Invalid overlap/trim: at least one unblended middle frame must remain.")
    device = torch.device(device or images.device)
    output_device = torch.device(output_device or images.device)
    count = length - k
    output = torch.empty((count, *images.shape[1:]), dtype=images.dtype, device=output_device)
    middle = length - 2 * k
    for offset in range(0, middle, chunk):
        if check_interrupt:
            check_interrupt()
        end = min(offset + chunk, middle)
        output[offset:end].copy_(images[a + k + offset:a + k + end].to(output_device))
        if on_progress:
            on_progress(end, count)
    if k:
        weights = blend_weights(k, device)
        for offset in range(0, k, chunk):
            if check_interrupt:
                check_interrupt()
            end = min(offset + chunk, k)
            tail = images[a + length - k + offset:a + length - k + end].to(device, dtype=torch.float32)
            head = images[a + offset:a + end].to(device, dtype=torch.float32)
            bridge = blend_frames(tail, head, weights[offset:end, None, None, None], blend_space)
            output[middle + offset:middle + end].copy_(bridge.to(output_device, dtype=images.dtype))
            if on_progress:
                on_progress(middle + end, count)
    return output


def _edges(frames):
    gray = frames[..., :min(3, frames.shape[-1])].mean(-1)
    dx = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


def _region_error(error, detail):
    """70% salience weighted pixels + 30% worst of an 8x8 region grid."""
    weights = 1 + 3 * (detail / (detail.mean((-2, -1), keepdim=True) + 1e-6)).clamp(max=20)
    weighted = (error * weights).sum((-2, -1)) / weights.sum((-2, -1))
    h, w = error.shape[-2:]
    tiles = F.adaptive_avg_pool2d(error.reshape(-1, 1, h, w), (min(8, h), min(8, w)))
    worst = tiles.flatten(1).amax(1).reshape(error.shape[:-2])
    return 0.7 * weighted + 0.3 * worst


def _temporal_peak(values):
    # Fixed max contribution prevents a long bridge diluting a bad junction.
    return 0.75 * values.amax(1) + 0.25 * values.mean(1)


def _local_excess(maps, values, context, changed):
    """Detect a quiet-region jump that a scalar motion allowance can conceal.

    Nearby motion can pass between regions, so either original side can support
    a region's reference. Scale that support to the quieter side's overall rate.
    """
    b, t, h, w = maps.shape
    tiles = F.adaptive_avg_pool2d(maps.reshape(-1, 1, h, w), (min(8, h), min(8, w))).reshape(b, t, -1)
    left, right = values[:, :context].amax(1), values[:, -context:].amax(1)
    factor = torch.minimum(left, right) / torch.maximum(left, right).clamp_min(1e-6)
    reference = torch.maximum(tiles[:, :context].amax(1), tiles[:, -context:].amax(1)) * factor[:, None]
    excess = (tiles[:, changed] - 1.5 * reference[:, None]).clamp_min(0).amax(-1)
    return 0.3 * _temporal_peak(excess)


def _fade_representatives(candidates, scores, top_count):
    """Keep each fade length in a small search, sampling larger ranges evenly."""
    best = {}
    for candidate, score in zip(candidates, scores):
        k = candidate.overlap
        if k not in best or (score, candidate) < (best[k][0], best[k][1]):
            best[k] = (score, candidate)
    levels = sorted(best)
    budget = 2 * top_count + 1
    if len(levels) > budget:
        levels = [levels[round(i * (len(levels) - 1) / (budget - 1))] for i in range(budget)]
    return [best[k][1] for k in levels]


def _discarded_frame_count(candidate):
    # Inclusive K>=2 drops the first head and last tail completely. K=2 has
    # no mixed frames, so it must be treated as a cut in preservation ties.
    return candidate.trim_start + candidate.trim_end + (2 if candidate.overlap >= 2 else 0)


def _proxy(images, size, device, chunk, check_interrupt, indices=None):
    h, w = images.shape[1:3]
    scale = min(1.0, size / max(h, w))
    shape = (max(1, round(h * scale)), max(1, round(w * scale)))
    count = len(images) if indices is None else len(indices)
    output = torch.empty((count, *shape, images.shape[-1]), device=device, dtype=torch.float32)
    for offset in range(0, count, chunk):
        if check_interrupt:
            check_interrupt()
        end = min(offset + chunk, count)
        if indices is None:
            part = images[offset:end]
        else:
            ix = torch.tensor(indices[offset:end], device=images.device)
            part = images.index_select(0, ix)
        if device.type == "cpu" and part.device.type != "cpu":
            # A CPU retry must not allocate resize/float temporaries on a full GPU.
            part = part.to("cpu")
        # Resize on the source device, before uploading full-resolution CPU data.
        part = F.interpolate(part.float().movedim(-1, 1), size=shape, mode="area").movedim(1, -1)
        output[offset:end].copy_(part.to(device))
    return output


def _boundary_indices(candidates, n):
    indices = set()
    for c in candidates:
        a, b, k = c.trim_start, c.trim_end, c.overlap
        length = n - a - b
        middle, cycle = length - 2 * k, length - k
        for pos in range(middle - 3 if k else length - 3, middle + k + 3 if k else length + 3):
            pos %= cycle
            if k and pos >= middle:
                j = pos - middle
                indices.update((a + length - k + j, a + j))
            else:
                indices.add(a + k + pos)
    return sorted(indices)


def _match_pair_tone(tail, head):
    """Compare structure at shared contrast, without counting a tone shift as ghosts.

    This affects scoring only. Rendered pixels retain their original color.
    Using the lower contrast avoids amplifying noise in nearly flat images.
    """
    channels = min(3, tail.shape[-1])
    t, h = tail[..., :channels], head[..., :channels]
    t_mean, h_mean = t.mean((-3, -2), keepdim=True), h.mean((-3, -2), keepdim=True)
    t_std, h_std = t.std((-3, -2), correction=0, keepdim=True), h.std((-3, -2), correction=0, keepdim=True)
    shared = torch.minimum(t_std, h_std)
    t = (t - t_mean) * (shared / t_std.clamp_min(1e-6))
    h = (h - h_mean) * (shared / h_std.clamp_min(1e-6))
    if tail.shape[-1] > channels:
        t, h = torch.cat((t, tail[..., channels:]), -1), torch.cat((h, head[..., channels:]), -1)
    return t, h


def _score_batch(frames, edges, candidates, n, options, lookup=None):
    """Candidates in a batch share K, allowing one gathered transition tensor."""
    device = frames.device
    params = torch.tensor([(c.trim_start, c.trim_end, c.overlap) for c in candidates], device=device)
    a, b, _ = params.unbind(1)
    k = candidates[0].overlap
    length = n - a - b
    middle, count = length - 2 * k, length - k
    offsets = torch.arange(-3, k + 3 if k else 3, device=device)
    positions = ((middle if k else length)[:, None] + offsets) % count[:, None]
    j = (positions - middle[:, None]).clamp(min=0, max=max(k - 1, 0))
    raw_ix = a[:, None] + k + positions.clamp(max=(middle - 1)[:, None])
    tail_ix = a[:, None] + length[:, None] - k + j if k else raw_ix
    head_ix = a[:, None] + j if k else raw_ix
    if lookup is not None:
        raw_ix, tail_ix, head_ix = lookup[raw_ix], lookup[tail_ix], lookup[head_ix]
    raw = frames[raw_ix]
    if k:
        tail, head = frames[tail_ix], frames[head_ix]
        alpha = blend_weights(k, device)[j][..., None, None, None]
        bridge = blend_frames(tail, head, alpha, options.blend_space)
        sequence = torch.where((positions >= middle[:, None])[..., None, None, None], bridge, raw)
    else:
        sequence = raw
    detail = _edges(sequence)
    delta = sequence[:, 1:] - sequence[:, :-1]
    pair_detail = torch.maximum(detail[:, 1:], detail[:, :-1]) + delta.abs().mean(-1)
    step = _region_error(delta.abs().mean(-1), pair_detail)
    # Interior context gives acceleration samples on both sides of each junction.
    acceleration = delta[:, 1:] - delta[:, :-1]
    accel = _region_error(acceleration.abs().mean(-1), pair_detail[:, 1:])
    exposure = delta[..., :min(3, frames.shape[-1])].mean((-3, -2)).abs().mean(-1)
    frame_contrast = sequence[..., :min(3, frames.shape[-1])].std((-3, -2), correction=0)
    contrast_steps = (frame_contrast[:, 1:] - frame_contrast[:, :-1]).abs().mean(-1)
    scale = options.fps / 24.0
    # A camera move in the middle of the clip must not excuse a cut between
    # settled endpoints. Reference actual original motion next to each junction.
    def transition_reference(values):
        left = values[:, :2].amax(1)
        right = values[:, -2:].amax(1)
        # Use the quieter side throughout. Interpolating these scalar tolerances
        # would excuse a cut inside a K=2 bridge when the head starts moving.
        reference = torch.minimum(left, right)[:, None]
        return reference

    step_reference = transition_reference(step)
    color_reference = transition_reference(exposure)
    contrast_reference = transition_reference(contrast_steps)
    # For inclusive endpoints, two acceleration samples at each end are still
    # original. K=0/1 only have one untouched sample at each end. Never include
    # a seam-affected sample in its own tolerance.
    left_accel = accel[:, :2].amax(1) if k >= 2 else accel[:, 0]
    right_accel = accel[:, -2:].amax(1) if k >= 2 else accel[:, -1]
    accel_reference = torch.minimum(left_accel, right_accel)[:, None]
    # Inclusive K>=2 endpoints leave the incoming/outgoing source pairs intact;
    # only internal bridge pairs are newly constructed. K=0/1 change both links.
    changed_steps = step[:, 3:-3] if k >= 2 else step[:, 2:-2]
    changed_exposure = exposure[:, 3:-3] if k >= 2 else exposure[:, 2:-2]
    changed_contrast = contrast_steps[:, 3:-3] if k >= 2 else contrast_steps[:, 2:-2]
    appearance = _temporal_peak((changed_steps - step_reference * 1.5).clamp_min(0)) * scale
    color = _temporal_peak((changed_exposure - color_reference * 1.5).clamp_min(0)) * scale
    contrast = _temporal_peak((changed_contrast - contrast_reference * 1.5).clamp_min(0)) * scale
    changed_accel = accel[:, 2:-2] if k >= 2 else accel[:, 1:-1]
    motion = _temporal_peak((changed_accel - accel_reference * 1.5).clamp_min(0)) * scale ** 2
    pair_slice = slice(3, -3) if k >= 2 else slice(2, -2)
    local_appearance = _local_excess(delta.abs().mean(-1), step, 2, pair_slice) * scale
    appearance = torch.maximum(appearance, local_appearance)
    accel_slice = slice(2, -2) if k >= 2 else slice(1, -1)
    local_motion = _local_excess(acceleration.abs().mean(-1), accel, 2 if k >= 2 else 1, accel_slice) * scale ** 2
    motion = torch.maximum(motion, local_motion)
    smoothness = (step[:, 1:] - step[:, :-1]).abs()[:, 1:-1].amax(1) * scale
    ghost = torch.zeros(len(candidates), device=device)
    if k:
        # Measure overlapping incompatible structure, weighted by actual mixing.
        tail_ix = a[:, None] + length[:, None] - k + torch.arange(k, device=device)
        head_ix = a[:, None] + torch.arange(k, device=device)
        if lookup is not None:
            tail_ix, head_ix = lookup[tail_ix], lookup[head_ix]
        t, h = frames[tail_ix], frames[head_ix]
        e = torch.maximum(edges[tail_ix], edges[head_ix])
        t, h = _match_pair_tone(t, h)
        mismatch = _region_error((t - h).square().mean(-1), e)
        # Edges live in normalized image coordinates, not raw proxy pixel units.
        edge_error = ((_edges(t) - _edges(h)).abs() * max(frames.shape[1:3]) / 96).clamp(max=1)
        edge_error = _region_error(edge_error.square(), e)
        w = blend_weights(k, device)[None, :]
        # Energy of the weaker mixed contour. The old 4*w*(1-w) amplitude
        # charged a faint overlay almost the full endpoint difference.
        ghost = _temporal_peak((mismatch + 0.25 * edge_error) * torch.minimum(w, 1 - w).square())
    duration = (n - count).float() / n
    fade = torch.full_like(duration, k / n)
    # Retain the existing report key; the endpoint-only objective no longer uses
    # whole-clip activity. Retention constraints and duration still protect length.
    activity_loss = torch.zeros_like(duration)
    metrics = torch.stack((appearance, color, motion, smoothness, ghost, duration, fade, activity_loss, contrast), 1)
    weights = torch.tensor([options.appearance_weight, options.exposure_weight, options.motion_weight,
                            options.smoothness_weight, options.ghosting_weight, options.duration_weight,
                            options.fade_weight, options.duration_weight, options.exposure_weight], device=device)
    return (metrics * weights).sum(1), metrics


def _score_candidates(frames, candidates, n, options, chunk, check_interrupt, on_progress=None, lookup=None):
    edges = _edges(frames)
    ordered = sorted(enumerate(candidates), key=lambda item: item[1].overlap)
    scores = torch.empty(len(candidates), device=frames.device)
    metrics = torch.empty((len(candidates), len(METRICS)), device=frames.device)
    done = 0
    for _, group in groupby(ordered, key=lambda item: item[1].overlap):
        group = list(group)
        for offset in range(0, len(group), chunk):
            if check_interrupt:
                check_interrupt()
            batch = group[offset:offset + chunk]
            s, m = _score_batch(frames, edges, [c for _, c in batch], n, options, lookup)
            indices = torch.tensor([i for i, _ in batch], device=frames.device)
            scores[indices], metrics[indices] = s, m
            done += len(batch)
            if on_progress:
                on_progress(done, len(candidates))
    return scores, metrics


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def is_cuda_compatibility_error(exc):
    """Narrow allowlist; programming errors and interruption always propagate."""
    message = str(exc).lower()
    return any(text in message for text in ("no kernel image is available", "invalid device function",
               "cuda driver version is insufficient", "found no nvidia driver", "not compiled with cuda",
               "cuda-capable device(s) is/are busy or unavailable"))


def cuda_smoke_test(device):
    x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    with torch.autocast(device_type=device.type, enabled=False):
        value = (x.square().sum() + (x @ x.T).sum()).item()
    return {"passed": value == 1640.0, "value": value, "operation": "fp32 square, reduction, matmul"}


def _memory_chunks(images, options, device):
    # Use a conservative share of currently FREE memory, leaving loaded models alone.
    budget = 128 * 1024 ** 2
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(budget, max(1024 ** 2, int(free * 0.1)))
    size = QUALITY[options.quality][0]
    h, w, channels = images.shape[1:]
    scale = min(1, size / max(h, w))
    pixels = max(1, round(h * scale)) * max(1, round(w * scale))
    scoring_bytes = (min(options.max_fade, (len(images) - 1) // 2) + 6) * pixels * channels * 4 * 16
    render_bytes = h * w * channels * 4 * (12 if options.blend_space == "linear" else 6)
    return min(options.candidate_chunk, max(1, budget // scoring_bytes)), min(options.render_chunk, max(1, budget // render_bytes))


def select_device(mode, preferred_device=None):
    diagnostics = {"torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "gpu_name": None, "capability": None, "smoke_test": {"passed": None, "reason": "not requested"},
                   "fallback_reason": None}
    preferred = torch.device(preferred_device) if preferred_device is not None else None
    if mode == "cpu" or (mode == "auto" and preferred is not None and preferred.type != "cuda"):
        if mode == "auto" and preferred.type != "cpu":
            diagnostics["fallback_reason"] = f"Configured {preferred.type} backend is outside this node's CPU/CUDA backend."
        return torch.device("cpu"), diagnostics
    if not torch.cuda.is_available():
        diagnostics["fallback_reason"] = "CUDA unavailable in the installed PyTorch/driver."
        if mode == "cuda":
            raise RuntimeError("Forced CUDA cannot run: CUDA is unavailable. Check ComfyUI's PyTorch and NVIDIA driver, or select cpu.")
        return torch.device("cpu"), diagnostics
    device = preferred if preferred is not None and preferred.type == "cuda" else torch.device("cuda")
    try:
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        diagnostics["gpu_name"] = torch.cuda.get_device_name(device)
        diagnostics["capability"] = list(torch.cuda.get_device_capability(device))
        diagnostics["smoke_test"] = cuda_smoke_test(device)
        if not diagnostics["smoke_test"]["passed"]:
            raise RuntimeError("CUDA smoke test produced an incorrect result.")
    except (RuntimeError, AssertionError) as exc:
        if not isinstance(exc, torch.cuda.OutOfMemoryError) and not is_cuda_compatibility_error(exc):
            raise
        diagnostics["smoke_test"] = {"passed": False, "reason": str(exc)}
        diagnostics["fallback_reason"] = f"CUDA smoke test failed: {exc}"
        if mode == "cuda":
            raise RuntimeError("Forced CUDA smoke test failed. Install a compatible ComfyUI PyTorch/driver or select cpu. " + str(exc)) from exc
        return torch.device("cpu"), diagnostics
    return device, diagnostics


@torch.inference_mode()
def _search(images, options, device, candidates, chunk, render_chunk, output_device, check_interrupt, on_progress):
    n = len(images)
    small_size, refine_size, top_count = QUALITY[options.quality]
    timings = {}
    _sync(device)
    start = time.perf_counter()
    indices = _boundary_indices(candidates, n)
    proxy = _proxy(images, small_size, device, render_chunk, check_interrupt, indices)
    lookup = torch.full((n,), -1, dtype=torch.long, device=device)
    lookup[torch.tensor(indices, device=device)] = torch.arange(len(indices), device=device)
    _sync(device)
    timings["proxy_and_transfer_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    def search_progress(done, total):
        if on_progress:
            on_progress(10 + int(55 * done / total), 100)
    scores, _ = _score_candidates(proxy, candidates, n, options, chunk, check_interrupt, search_progress, lookup)
    # A single batched synchronization for shortlist selection, never per candidate.
    score_values = scores.cpu().tolist()
    rank = sorted(range(len(candidates)), key=lambda i: (score_values[i], candidates[i]))
    shortlist = [candidates[i] for i in rank[:top_count]]
    identity = LoopCandidate()
    if identity in candidates and identity not in shortlist:
        shortlist.append(identity)
    # Keep the cheapest satisfactory repairs in addition to numerical top scores.
    satisfactory = [i for i in rank if score_values[i] <= options.satisfactory_score]
    satisfactory.sort(key=lambda i: (sum(asdict(candidates[i]).values()), candidates[i].overlap, candidates[i]))
    for i in satisfactory[:top_count]:
        if candidates[i] not in shortlist:
            shortlist.append(candidates[i])
    for candidate in _fade_representatives(candidates, score_values, top_count):
        if candidate not in shortlist:
            shortlist.append(candidate)
    del scores, proxy, lookup
    _sync(device)
    timings["search_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    indices = _boundary_indices(shortlist, n)
    refined = _proxy(images, refine_size, device, render_chunk, check_interrupt, indices)
    lookup = torch.full((n,), -1, dtype=torch.long, device=device)
    lookup[torch.tensor(indices, device=device)] = torch.arange(len(indices), device=device)
    refined_scores, metrics = _score_candidates(refined, shortlist, n, options,
                                                max(1, chunk // 4), check_interrupt, lookup=lookup)
    rows = torch.cat((refined_scores[:, None], metrics), 1).cpu().tolist()
    _sync(device)
    timings["refinement_and_transfer_seconds"] = time.perf_counter() - start
    ranked = sorted(range(len(shortlist)), key=lambda i: (rows[i][0], shortlist[i]))
    valid = [i for i in ranked if rows[i][0] <= options.satisfactory_score]
    identity_index = shortlist.index(identity) if identity in shortlist else None
    preserve_identity = (identity_index is not None
                         and rows[identity_index][0] <= options.satisfactory_score
                         and rows[identity_index][0] <= rows[ranked[0]][0] + options.tie_tolerance)
    if preserve_identity:
        winner = shortlist.index(identity)
        selection = "unchanged_satisfactory"
    elif valid:
        repairs = [i for i in valid if i != identity_index]
        # At equal retained duration, prefer blending over discarding more whole
        # source frames. Smaller K alone must not give a cut an automatic win.
        winner = min(repairs, key=lambda i: (sum(asdict(shortlist[i]).values()),
                                            _discarded_frame_count(shortlist[i]), rows[i][0], shortlist[i]))
        selection = "shortest_satisfactory_repair"
    else:
        winner = ranked[0]
        selection = "best_bounded_candidate"
    selected = shortlist[winner]
    alternatives = [{**asdict(shortlist[i]), "score": rows[i][0]} for i in ranked if i != winner][:5]
    tied = [row for row in alternatives if abs(row["score"] - rows[winner][0]) <= options.tie_tolerance]
    diagnostics = []
    if identity_index in valid and not preserve_identity:
        diagnostics.append("unchanged passed threshold, but a meaningfully better refined repair exists")
    if not valid:
        diagnostics.append("low_confidence: no refined candidate met the satisfactory threshold")
    if tied:
        diagnostics.append("uncertain: nearly tied refined candidates")
    fade_comparison = []
    for k in sorted(set(c.overlap for c in shortlist)):
        i = min((i for i, c in enumerate(shortlist) if c.overlap == k), key=lambda i: (rows[i][0], shortlist[i]))
        fade_comparison.append({**asdict(shortlist[i]), "score": rows[i][0],
                                "metrics": dict(zip(METRICS, rows[i][1:]))})
    report = {"selected": asdict(selected), "selection": selection, "score": rows[winner][0],
              "selection_policy": "shortest satisfactory repair, then fewest fully discarded source frames",
              "fully_discarded_source_frames": _discarded_frame_count(selected),
              "fade_comparison": fade_comparison,
              "metrics": dict(zip(METRICS, rows[winner][1:])), "alternatives": alternatives, "nearly_tied": tied,
              "confidence": "low" if not valid else ("uncertain" if tied else "heuristic_satisfactory"),
              "diagnostics": diagnostics, "candidate_count": len(candidates), "refined_count": len(shortlist),
              "input_frames": n, "output_frames": n - selected.trim_start - selected.trim_end - selected.overlap,
              "fps": options.fps, "input_duration_seconds": n / options.fps,
              "blend_space": options.blend_space, "proxy_long_edge": small_size, "refine_long_edge": refine_size,
              "options": asdict(options), "objective_version": 4, "scoring_scope": "candidate boundary windows only",
              "optical_flow": "not used; temporal difference heuristic"}
    report["output_duration_seconds"] = report["output_frames"] / options.fps
    del refined, refined_scores, metrics, lookup
    if on_progress:
        on_progress(85, 100)
    start = time.perf_counter()
    def render_progress(done, total):
        if on_progress:
            on_progress(85 + int(15 * done / total), 100)
    result = render_cycle(images, selected, options.blend_space, device, output_device, render_chunk,
                          check_interrupt, render_progress)
    _sync(device)
    _sync(output_device)
    timings["render_and_transfer_seconds"] = time.perf_counter() - start
    report["timings"] = timings
    return result, report


@torch.inference_mode()
def optimize_loop(images, options=None, preferred_device=None, output_device=None, check_interrupt=None, on_progress=None):
    """Return (IMAGE, JSON-serializable report). Standalone output defaults to CPU.

    Auto retries CUDA OOM twice with halved chunks, then CPU. Forced CUDA retries
    twice but never silently switches backend. Non-memory/non-compatibility errors
    and caller cancellation are never converted into fallback.
    """
    options = options or LoopOptions()
    validate_options(options)
    started = time.perf_counter()
    _validate_image_structure(images)
    candidates = enumerate_candidates(len(images), options)
    init_start = time.perf_counter()
    device, diagnostics = select_device(options.device, preferred_device)
    _sync(device)
    init_seconds = time.perf_counter() - init_start
    output_device = torch.device(output_device or "cpu")
    if device.type == "cpu" and output_device.type == "cuda" and diagnostics["fallback_reason"]:
        raise RuntimeError("CUDA fallback needs CPU output; ComfyUI's GPU-only intermediate policy requires GPU output. Free VRAM or disable --gpu-only.")
    chunk, render_chunk = _memory_chunks(images, options, device)
    attempts = []
    for attempt in range(4):
        failure = None
        try:
            validation_device = "cpu" if device.type == "cpu" else images.device
            validate_images(images, render_chunk, check_interrupt, validation_device)
            with torch.autocast(device_type=device.type, enabled=False):
                result, report = _search(images, options, device, candidates, chunk, render_chunk, output_device,
                                         check_interrupt, on_progress)
        except torch.cuda.OutOfMemoryError as exc:
            if device.type != "cuda":
                raise
            failure = f"CUDA OOM: {exc}"
        except RuntimeError as exc:
            if device.type != "cuda" or not is_cuda_compatibility_error(exc):
                raise
            failure = f"CUDA compatibility failure: {exc}"
            if options.device == "cuda":
                raise RuntimeError("Forced CUDA cannot execute this PyTorch build. Check the driver/build or select cpu. " + str(exc)) from exc
        if failure is None:
            report["backend"] = {**diagnostics, "selected_device": str(device), "output_device": str(output_device),
                                 "attempts": attempts, "candidate_chunk": chunk, "render_chunk": render_chunk}
            report["timings"]["device_initialization_seconds"] = init_seconds
            report["timings"]["end_to_end_seconds"] = time.perf_counter() - started
            return result, report
        # Failed _search has unwound; no exception traceback retaining its buffers.
        attempts.append({"device": str(device), "candidate_chunk": chunk, "render_chunk": render_chunk, "reason": failure})
        if check_interrupt:
            check_interrupt()
        chunk, render_chunk = max(1, chunk // 2), max(1, render_chunk // 2)
        if "compatibility" in failure or attempt >= 2:
            if options.device == "cuda":
                raise RuntimeError("Forced CUDA exhausted two reduced-chunk OOM retries. Free VRAM or select cpu.")
            # CPU output is essential for a safe fallback; obey explicit downstream device.
            if output_device.type != "cpu":
                raise RuntimeError("CUDA fallback needs CPU output; ComfyUI's GPU-only intermediate policy requires GPU output. Free VRAM or disable --gpu-only.")
            diagnostics["fallback_reason"] = failure
            device = torch.device("cpu")
    raise RuntimeError("Loop optimization exhausted its bounded retry policy.")
