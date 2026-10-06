"""Sample-aligned audio construction for an already selected video loop.

No codecs, ComfyUI imports, resampling or change of playback speed. Audio is
assumed to start at video frame zero. See docs/auto_seamless_loop.md.
"""

import math
from numbers import Integral, Real

import torch


SAMPLE_CHUNK = 65536


def validate_loop_audio(audio, check_interrupt=None):
    """Validate optional ComfyUI AUDIO; return waveform/rate or None.

    Batches and channels are preserved. Check finite samples on CPU in bounded
    chunks, independently of the device used for video scoring.
    """
    if audio is None:
        return None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio must be a ComfyUI AUDIO dictionary with waveform and sample_rate.")
    waveform, rate = audio["waveform"], audio["sample_rate"]
    if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 3
            or waveform.shape[0] < 1 or waveform.shape[1] < 1 or not waveform.is_floating_point()):
        raise ValueError("Audio waveform must be a floating-point tensor [batch, channels, samples].")
    if isinstance(rate, bool) or not isinstance(rate, Integral) or not 1 <= rate <= 384000:
        raise ValueError("Audio sample_rate must be an integer between 1 and 384000.")
    for start in range(0, waveform.shape[-1], SAMPLE_CHUNK):
        if check_interrupt:
            check_interrupt()
        if not torch.isfinite(waveform[..., start:start + SAMPLE_CHUNK].to("cpu")).all():
            raise ValueError("Audio contains NaN or infinite samples.")
    return waveform, int(rate)


@torch.inference_mode()
def render_loop_audio(audio, frame_count, fps, candidate, check_interrupt=None):
    """Return (optional AUDIO, timing report) for one video cycle, on CPU.

    Copy the middle starting at frame a+K, then mix tail/head audio into a
    bridge. Source offsets and total duration round to the nearest sample;
    bridge length is the remainder, preventing accumulated rounding drift.
    Continuous linear audio gains match video weights at bridge frame centers.
    K=1 ramps across its single frame, reaching the video's 1/2 mix at its center.
    Short input audio is read as silence; excess input audio is discarded.
    """
    info = validate_loop_audio(audio, check_interrupt)
    if info is None:
        return None, {"present": False}
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count < 1:
        raise ValueError("Audio loop frame_count must be a positive integer.")
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Audio loop fps must be finite and positive.")
    a, b, k = candidate.trim_start, candidate.trim_end, candidate.overlap
    for value in (a, b, k):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("Audio loop trims and overlap must be nonnegative frame counts.")
    length = frame_count - a - b
    if length < 1 or (k and length < 2 * k + 1):
        raise ValueError("Audio loop trims/overlap must leave a valid video cycle.")
    waveform, rate = info
    samples_per_frame = rate / fps
    source_samples = round(frame_count * samples_per_frame)
    output_samples = round((length - k) * samples_per_frame)
    if output_samples < 1:
        raise ValueError("Audio sample rate is too low for the selected video cycle duration.")
    middle_samples = round((length - 2 * k) * samples_per_frame) if k else output_samples
    bridge_samples = output_samples - middle_samples
    output = torch.empty((*waveform.shape[:-1], output_samples), dtype=waveform.dtype, device="cpu")

    def read(start, count):
        # Bound by the original video duration as well as the waveform: codec
        # padding after the last input frame must not become part of the loop.
        available = min(count, max(0, min(waveform.shape[-1], source_samples) - start))
        part = torch.zeros((*waveform.shape[:-1], count), dtype=waveform.dtype, device="cpu")
        if available:
            part[..., :available].copy_(waveform[..., start:start + available].to("cpu"))
        return part

    middle_start = round((a + k) * samples_per_frame)
    for offset in range(0, middle_samples, SAMPLE_CHUNK):
        if check_interrupt:
            check_interrupt()
        count = min(SAMPLE_CHUNK, middle_samples - offset)
        output[..., offset:offset + count].copy_(read(middle_start + offset, count))
    head_start = round(a * samples_per_frame)
    tail_start = round((frame_count - b - k) * samples_per_frame)
    for offset in range(0, bridge_samples, SAMPLE_CHUNK):
        if check_interrupt:
            check_interrupt()
        count = min(SAMPLE_CHUNK, bridge_samples - offset)
        # FP32 avoids low-precision ramps; FP64 input keeps its precision.
        dtype = torch.float64 if waveform.dtype == torch.float64 else torch.float32
        positions = torch.arange(offset, offset + count, dtype=torch.float64)
        if k == 1:
            weights = positions / max(1, bridge_samples - 1)
            if bridge_samples == 1:
                weights.fill_(0.5)
        else:
            weights = ((positions / samples_per_frame - 0.5) / (k - 1)).clamp_(0, 1)
        tail = read(tail_start + offset, count).to(dtype)
        head = read(head_start + offset, count).to(dtype)
        # Convex linear gain avoids amplitude growth for correlated sound.
        mixed = tail * (1 - weights.to(dtype)) + head * weights.to(dtype)
        output[..., middle_samples + offset:middle_samples + offset + count].copy_(mixed)
    report = {
        "present": True,
        "sample_rate": rate,
        "batch_size": waveform.shape[0],
        "channels": waveform.shape[1],
        "input_samples": waveform.shape[-1],
        "output_samples": output_samples,
        "output_duration_seconds": output_samples / rate,
        "video_duration_seconds": (length - k) / fps,
        "duration_rounding_error_seconds": output_samples / rate - (length - k) / fps,
        "source_start_sample": middle_start,
        "middle_samples": middle_samples,
        "bridge_samples": bridge_samples,
        "padded_source_samples": max(0, source_samples - waveform.shape[-1]),
        "discarded_samples_after_source_video": max(0, waveform.shape[-1] - source_samples),
        "crossfade": "continuous linear amplitude" if k else "none",
    }
    return {"waveform": output, "sample_rate": rate}, report
