"""Pass through ComfyUI AUDIO or construct sample-aligned silence."""

import math
from numbers import Integral, Real

import torch


SILENCE_SAMPLE_RATE = 48000


def silence_sample_count(frame_count, fps):
    """Convert a positive frame duration to the nearest audio sample."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer.")
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number.")
    samples = frame_count / fps * SILENCE_SAMPLE_RATE
    if not math.isfinite(samples):
        raise ValueError("Requested audio duration is too large.")
    return max(1, round(samples))


def resolve_audio_fallback(audio, frame_count, fps):
    """Keep populated AUDIO unchanged; missing/empty audio becomes CPU silence.

    Silence in an existing waveform counts as audio. Frame timing only controls
    the fallback; it never trims, pads, resamples or copies populated audio.
    """
    if audio is not None:
        if not isinstance(audio, dict):
            raise ValueError("audio must be a ComfyUI AUDIO dictionary or None.")
        waveform = audio.get("waveform")
        if waveform is not None:
            if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 3
                    or waveform.shape[0] < 1 or waveform.shape[1] < 1
                    or not waveform.is_floating_point()):
                raise ValueError("Audio waveform must be a floating-point tensor [batch, channels, samples].")
            if waveform.shape[-1] > 0:
                rate = audio.get("sample_rate")
                if isinstance(rate, bool) or not isinstance(rate, Integral) or rate <= 0:
                    raise ValueError("Audio sample_rate must be a positive integer.")
                return audio

    samples = silence_sample_count(frame_count, fps)
    return {
        "waveform": torch.zeros((1, 2, samples), dtype=torch.float32, device="cpu"),
        "sample_rate": SILENCE_SAMPLE_RATE,
    }
