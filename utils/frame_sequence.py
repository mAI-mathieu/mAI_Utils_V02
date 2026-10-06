"""Frame-range calculations, trimming, and loop fades without ComfyUI."""

from numbers import Integral


TRIM_MODES = ("start", "end", "both ends")


def get_trim_bounds(frame_count, trim_mode, trim_amount):
    """Return an exclusive-end range, requiring at least one retained frame."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count < 1:
        raise ValueError("The frame sequence must contain at least one frame.")
    if trim_mode not in TRIM_MODES:
        raise ValueError(f"Unknown trim mode: {trim_mode}")
    if isinstance(trim_amount, bool) or not isinstance(trim_amount, Integral) or trim_amount < 0:
        raise ValueError("Trim amount must be a non-negative integer.")

    start = trim_amount if trim_mode in ("start", "both ends") else 0
    end = frame_count - trim_amount if trim_mode in ("end", "both ends") else frame_count
    if start >= end:
        raise ValueError(
            f"Cannot trim {trim_amount} frame(s) from {trim_mode} of a "
            f"{frame_count}-frame sequence; at least one frame must remain."
        )
    return start, end


def trim_frame_sequence(frames, trim_mode, trim_amount):
    """Trim a batch or a list of frames/batches and return one IMAGE batch."""
    batches = _get_frame_batches(frames)
    start, end = get_trim_bounds(
        sum(batch.shape[0] for batch in batches), trim_mode, trim_amount
    )
    return _slice_frame_batches(batches, start, end)


def get_independent_trim_bounds(frame_count, trim_start, trim_end):
    """Return an exclusive-end range using separate counts for each end."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count < 1:
        raise ValueError("The frame sequence must contain at least one frame.")
    for name, amount in (("trim_start", trim_start), ("trim_end", trim_end)):
        if isinstance(amount, bool) or not isinstance(amount, Integral) or amount < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    end = frame_count - trim_end
    if trim_start >= end:
        raise ValueError(
            f"Cannot trim {trim_start} frame(s) from the start and {trim_end} "
            f"from the end of a {frame_count}-frame sequence; at least one frame must remain."
        )
    return trim_start, end


def trim_frame_sequence_ends(frames, trim_start=0, trim_end=0):
    """Trim independent start/end counts and return one ordered IMAGE batch."""
    batches = _get_frame_batches(frames)
    start, end = get_independent_trim_bounds(
        sum(batch.shape[0] for batch in batches), trim_start, trim_end
    )
    return _slice_frame_batches(batches, start, end)


def append_loop_fade(frames, fade_frames=24):
    """Append exactly fade_frames images, blending the last frame into the first.

    The original sequence is preserved. Appended frame i uses first-frame opacity
    i / fade_frames (i starts at 1), so the last appended frame is the first image.
    """
    import torch

    if isinstance(fade_frames, bool) or not isinstance(fade_frames, Integral) or fade_frames < 0:
        raise ValueError("fade_frames must be a non-negative integer.")
    batches = _get_frame_batches(frames)
    if not batches[0].is_floating_point():
        raise ValueError("Loop fade requires floating-point IMAGE tensors.")
    if fade_frames == 0:
        return batches[0] if len(batches) == 1 else torch.cat(batches, dim=0)

    first = batches[0][:1]
    last = batches[-1][-1:]
    # Calculate weights in at least float32 before converting to the IMAGE dtype.
    weight_dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
    weights = torch.arange(1, fade_frames + 1, device=first.device, dtype=weight_dtype)
    weights = (weights / fade_frames).to(first.dtype).reshape(-1, 1, 1, 1)
    fade = torch.lerp(last, first, weights)
    # Avoid interpolation roundoff at the loop endpoint.
    fade[-1].copy_(first[0])
    return torch.cat([*batches, fade], dim=0)


def _get_frame_batches(frames):
    """Validate compatible IMAGE tensors and normalize individual HWC frames."""
    import torch

    batches = list(frames) if isinstance(frames, (list, tuple)) else [frames]
    if not batches:
        raise ValueError("The frame sequence must contain at least one frame.")

    for index, batch in enumerate(batches):
        if not isinstance(batch, torch.Tensor) or batch.ndim not in (3, 4):
            raise ValueError(
                f"Frame sequence item {index} must be an IMAGE tensor with shape "
                "[frames, height, width, channels] or [height, width, channels]."
            )
        if any(size < 1 for size in batch.shape):
            raise ValueError(f"Frame sequence item {index} has empty dimensions.")
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
            batches[index] = batch
        first = batches[0]
        if batch.shape[1:] != first.shape[1:]:
            raise ValueError("All frames must have the same height, width, and channels.")
        if batch.dtype != first.dtype or batch.device != first.device:
            raise ValueError("All frames must have the same tensor dtype and device.")

    return batches


def _slice_frame_batches(batches, start, end):
    import torch

    # Slice before concatenating so discarded frames do not need a new allocation.
    retained = []
    offset = 0
    for batch in batches:
        local_start = max(0, start - offset)
        local_end = min(batch.shape[0], end - offset)
        if local_start < local_end:
            retained.append(batch[local_start:local_end])
        offset += batch.shape[0]
    return retained[0] if len(retained) == 1 else torch.cat(retained, dim=0)
