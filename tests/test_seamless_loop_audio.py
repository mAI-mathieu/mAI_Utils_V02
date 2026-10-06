"""Audio timing ground truths independent of ComfyUI or video codecs."""

import json

import pytest
import torch

from nodes.auto_seamless_loop import MAIAutoSeamlessLoop
from utils.seamless_loop import LoopCandidate
from utils.seamless_loop_audio import render_loop_audio, validate_loop_audio


def audio(waveform=None, rate=32):
    if waveform is None:
        waveform = torch.arange(96, dtype=torch.float32).reshape(1, 1, -1) / 100
    return {"waveform": waveform, "sample_rate": rate}


def test_absent_audio_is_absent_output():
    result, report = render_loop_audio(None, 12, 4, LoopCandidate())
    assert result is None and report == {"present": False}


@pytest.mark.parametrize("candidate", [LoopCandidate(), LoopCandidate(1, 2, 0)])
def test_no_fade_is_exact_crop_without_mutating_source(candidate):
    source = audio()
    original = source["waveform"].clone()
    result, report = render_loop_audio(source, 12, 4, candidate)
    start, stop = candidate.trim_start * 8, (12 - candidate.trim_end) * 8
    torch.testing.assert_close(result["waveform"], original[..., start:stop], rtol=0, atol=0)
    assert result["waveform"].data_ptr() != source["waveform"].data_ptr()
    assert result["sample_rate"] == 32
    assert report["bridge_samples"] == 0
    assert report["output_duration_seconds"] == (stop - start) / 32
    torch.testing.assert_close(source["waveform"], original, rtol=0, atol=0)


def test_positive_fade_rotates_middle_and_matches_video_frame_center_gains():
    source = audio()
    result, report = render_loop_audio(source, 12, 4, LoopCandidate(1, 2, 3))
    waveform = result["waveform"]
    # Trimmed frames 1..9; middle frames 4..6, bridge mixes 7..9 with 1..3.
    assert waveform.shape == (1, 1, 48)
    torch.testing.assert_close(waveform[..., :24], source["waveform"][..., 32:56])
    for j, gain in enumerate((0, 0.5, 1)):
        center = j * 8 + 4
        expected = (1 - gain) * source["waveform"][..., 56 + center] + gain * source["waveform"][..., 8 + center]
        torch.testing.assert_close(waveform[..., 24 + center], expected)
    # First/last gain plateaus keep the natural connection to the middle.
    torch.testing.assert_close(waveform[..., 24:28], source["waveform"][..., 56:60])
    torch.testing.assert_close(waveform[..., -4:], source["waveform"][..., 28:32])
    assert report["source_start_sample"] == 32 and report["bridge_samples"] == 24


def test_single_frame_fade_has_continuous_gain_and_correct_rotation():
    source = audio()
    waveform = render_loop_audio(source, 12, 4, LoopCandidate(1, 2, 1))[0]["waveform"]
    assert waveform.shape[-1] == 64
    torch.testing.assert_close(waveform[..., :56], source["waveform"][..., 16:72])
    torch.testing.assert_close(waveform[..., 56], source["waveform"][..., 72])
    torch.testing.assert_close(waveform[..., -1], source["waveform"][..., 15])
    # The center of the sampled ramp represents the video's midpoint mix.
    expected = (source["waveform"][..., 75] + source["waveform"][..., 11]
                + source["waveform"][..., 76] + source["waveform"][..., 12]) / 4
    torch.testing.assert_close(waveform[..., 59:61].mean(-1), expected)


@pytest.mark.parametrize("fps", [24, 29.97, 30000 / 1001, 59.94])
@pytest.mark.parametrize("k", [0, 1, 4])
def test_fractional_samples_per_frame_have_no_cumulative_duration_drift(fps, k):
    rate = 44100
    count = 121
    source = audio(torch.zeros(1, 2, round(count * rate / fps)), rate)
    waveform, report = render_loop_audio(source, count, fps, LoopCandidate(2, 3, k))
    duration = (count - 2 - 3 - k) / fps
    assert waveform["waveform"].shape[-1] == round(duration * rate)
    assert abs(report["duration_rounding_error_seconds"]) <= 0.50001 / rate
    assert report["middle_samples"] + report["bridge_samples"] == report["output_samples"]
    assert report["source_start_sample"] == round((2 + k) * rate / fps)


def test_short_audio_is_silence_padded_and_long_audio_is_truncated():
    short = audio(torch.ones(1, 1, 70))
    result, report = render_loop_audio(short, 12, 4, LoopCandidate(1, 0, 0))
    torch.testing.assert_close(result["waveform"][..., :62], torch.ones(1, 1, 62))
    assert not result["waveform"][..., 62:].any()
    assert report["padded_source_samples"] == 26
    long = audio(torch.arange(120, dtype=torch.float32).reshape(1, 1, -1))
    result, report = render_loop_audio(long, 12, 4, LoopCandidate())
    torch.testing.assert_close(result["waveform"], long["waveform"][..., :96])
    assert report["discarded_samples_after_source_video"] == 24


def test_silence_padding_is_applied_before_overlap_mix():
    source = audio(torch.ones(1, 1, 56))
    waveform, report = render_loop_audio(source, 12, 4, LoopCandidate(1, 2, 3))
    assert report["padded_source_samples"] == 40
    # Tail is entirely missing, head is valid: video-frame-center gains 0, .5, 1.
    torch.testing.assert_close(waveform["waveform"][..., [28, 36, 44]], torch.tensor([[[0, 0.5, 1]]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_batch_channels_dtype_and_linear_gain_are_preserved(dtype):
    # Identical sound in both regions retains its gain rather than doubling.
    source = audio(torch.full((2, 3, 96), 0.75, dtype=dtype))
    result, _ = render_loop_audio(source, 12, 4, LoopCandidate(1, 1, 4))
    assert result["waveform"].dtype == dtype and result["waveform"].device.type == "cpu"
    torch.testing.assert_close(result["waveform"], torch.full((2, 3, 48), 0.75, dtype=dtype))


def test_crossfade_is_continuous_across_processing_chunks():
    rate = 48000
    source = audio(torch.zeros(1, 1, 12 * rate), rate)
    source["waveform"][..., :4 * rate] = 1
    result, _ = render_loop_audio(source, 12, 1, LoopCandidate(0, 0, 4))
    bridge = result["waveform"][..., 4 * rate:]
    positions = torch.tensor([65535, 65536, 131071, 131072], dtype=torch.float64)
    expected = ((positions / rate - 0.5) / 3).float().reshape(1, 1, -1)
    torch.testing.assert_close(bridge[..., positions.long()], expected)


def test_empty_waveform_becomes_silence():
    result, report = render_loop_audio(audio(torch.zeros(1, 2, 0)), 12, 4, LoopCandidate(1, 1, 3))
    assert result["waveform"].shape == (1, 2, 56) and not result["waveform"].any()
    assert report["padded_source_samples"] == 96


@pytest.mark.parametrize("source", [
    {}, {"waveform": torch.zeros(1, 1, 8)},
    audio(torch.zeros(1, 8)), audio(torch.zeros(0, 1, 8)),
    audio(torch.zeros(1, 0, 8)), audio(torch.zeros(1, 1, 8, dtype=torch.int16)),
    audio(rate=True), audio(rate=0), audio(rate=44100.0), audio(rate=400000),
    audio(torch.full((1, 1, 8), float("nan"))), audio(torch.full((1, 1, 8), float("inf"))),
])
def test_invalid_audio_has_clear_error(source):
    with pytest.raises(ValueError, match="[Aa]udio"):
        validate_loop_audio(source)


@pytest.mark.parametrize("count,fps,candidate", [
    (0, 4, LoopCandidate()), (True, 4, LoopCandidate()),
    (12, 0, LoopCandidate()), (12, float("inf"), LoopCandidate()),
    (12, True, LoopCandidate()), (12, 4, LoopCandidate(-1, 0, 0)),
    (12, 4, LoopCandidate(6, 6, 0)), (12, 4, LoopCandidate(1, 2, 5)),
])
def test_invalid_video_timing_has_clear_error(count, fps, candidate):
    with pytest.raises(ValueError, match="Audio"):
        render_loop_audio(audio(), count, fps, candidate)


def test_cancellation_is_propagated():
    def interrupt():
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError, match="cancelled"):
        render_loop_audio(audio(), 12, 4, LoopCandidate(), check_interrupt=interrupt)


def test_node_audio_uses_the_selected_video_candidate():
    source = audio()
    result = MAIAutoSeamlessLoop().run(
        torch.zeros(12, 2, 3, 3), fps=4, device="cpu", quality="fast", audio=source,
        max_trim_start=1, max_trim_end=1, max_fade=3, min_retained_percent=50,
        manual_trim_start=1, manual_trim_end=1, manual_overlap=3)
    assert result[1:6] == (4.0, 7, 1, 1, 3)
    expected, _ = render_loop_audio(source, 12, 4, LoopCandidate(1, 1, 3))
    torch.testing.assert_close(result[7]["waveform"], expected["waveform"])
    report = json.loads(result[6])
    assert report["audio"]["output_duration_seconds"] == len(result[0]) / result[1]


def test_node_rejects_invalid_audio_before_video_search(monkeypatch):
    import nodes.auto_seamless_loop as node_module
    def forbidden(*args, **kwargs):
        raise AssertionError("Video search should not run for invalid audio")
    monkeypatch.setattr(node_module, "optimize_loop", forbidden)
    with pytest.raises(ValueError, match="sample_rate"):
        MAIAutoSeamlessLoop().run(torch.zeros(12, 2, 3, 3), audio=audio(rate=0))
