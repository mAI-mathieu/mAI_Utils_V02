import pytest
import torch

from utils.frame_sequence import (
    get_independent_trim_bounds,
    get_trim_bounds,
    trim_frame_sequence,
    trim_frame_sequence_ends,
)


@pytest.mark.parametrize(
    "mode,amount,expected",
    [("start", 2, (2, 10)), ("end", 2, (0, 8)), ("both ends", 2, (2, 8)),
     ("start", 0, (0, 10)), ("end", 0, (0, 10)), ("both ends", 0, (0, 10)),
     ("start", 9, (9, 10)), ("end", 9, (0, 1))],
)
def test_trim_bounds(mode, amount, expected):
    assert get_trim_bounds(10, mode, amount) == expected


@pytest.mark.parametrize(
    "count,mode,amount",
    [(0, "start", 0), (1, "start", 1), (10, "end", 11), (10, "both ends", 5),
     (10, "unknown", 0), (10, "start", -1), (10, "end", 1.5),
     (10, "start", True), (10, "start", "2")],
)
def test_invalid_trim_bounds(count, mode, amount):
    with pytest.raises(ValueError):
        get_trim_bounds(count, mode, amount)


@pytest.mark.parametrize("mode,start,end", [("start", 3, 10), ("end", 0, 7), ("both ends", 3, 7)])
@pytest.mark.parametrize("representation", ["batch", "frames", "single_batches", "chunks"])
def test_trim_preserves_exact_frames_across_list_boundaries(mode, start, end, representation):
    original = torch.arange(120, dtype=torch.float64).reshape(10, 2, 2, 3)
    before = original.clone()
    inputs = {
        "batch": original,
        "frames": list(original),
        "single_batches": list(original.split(1)),
        "chunks": [original[:2], original[2:6], original[6:]],
    }
    result = trim_frame_sequence(inputs[representation], mode, 3)
    assert torch.equal(result, original[start:end])
    assert result.dtype == original.dtype
    assert result.device == original.device
    assert torch.equal(original, before)


@pytest.mark.parametrize("mode", ["start", "end", "both ends"])
def test_zero_trim_keeps_single_frame(mode):
    frame = torch.rand(1, 2, 2, 3)
    assert torch.equal(trim_frame_sequence(frame, mode, 0), frame)


def test_single_remaining_frame_with_both_ends():
    frames = torch.arange(15).reshape(5, 1, 1, 3)
    assert torch.equal(trim_frame_sequence(frames, "both ends", 2), frames[2:3])


@pytest.mark.parametrize(
    "frames,message",
    [([], "at least one"), (None, "IMAGE tensor"), (["frame"], "IMAGE tensor"),
     (torch.zeros(2, 2), "IMAGE tensor"), (torch.zeros(0, 2, 2, 3), "empty dimensions"),
     ([torch.zeros(1, 2, 2, 3), torch.zeros(1, 3, 2, 3)], "same height"),
     ([torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 4)], "same height"),
     ([torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 3, dtype=torch.float64)], "dtype and device")],
)
def test_rejects_invalid_sequences(frames, message):
    with pytest.raises(ValueError, match=message):
        trim_frame_sequence(frames, "start", 0)
    with pytest.raises(ValueError, match=message):
        trim_frame_sequence_ends(frames, 0, 0)


@pytest.mark.parametrize(
    "count,start,end,expected",
    [(10, 0, 0, (0, 10)), (10, 2, 0, (2, 10)), (10, 0, 3, (0, 7)),
     (10, 2, 3, (2, 7)), (10, 9, 0, (9, 10)), (10, 0, 9, (0, 1)),
     (10, 4, 5, (4, 5)), (1, 0, 0, (0, 1))],
)
def test_independent_trim_bounds(count, start, end, expected):
    assert get_independent_trim_bounds(count, start, end) == expected


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "10"])
def test_independent_bounds_reject_invalid_frame_count(count):
    with pytest.raises(ValueError, match="at least one frame"):
        get_independent_trim_bounds(count, 0, 0)


@pytest.mark.parametrize("amount", [-1, True, 1.5, "2", None])
@pytest.mark.parametrize("setting", ["trim_start", "trim_end"])
def test_independent_bounds_reject_invalid_counts(amount, setting):
    counts = {"trim_start": 0, "trim_end": 0}
    counts[setting] = amount
    with pytest.raises(ValueError, match=f"{setting} must be a non-negative integer"):
        get_independent_trim_bounds(10, **counts)


@pytest.mark.parametrize("start,end", [(10, 0), (0, 10), (4, 6), (5, 7)])
def test_independent_bounds_reject_trimming_entire_sequence(start, end):
    with pytest.raises(ValueError, match="at least one frame must remain"):
        get_independent_trim_bounds(10, start, end)


@pytest.mark.parametrize("representation", ["batch", "frames", "single_batches", "chunks"])
@pytest.mark.parametrize("start,end", [(0, 0), (2, 0), (0, 3), (2, 3), (4, 5)])
def test_independent_trims_preserve_frames_across_batch_boundaries(representation, start, end):
    original = torch.arange(120, dtype=torch.float64).reshape(10, 2, 2, 3)
    before = original.clone()
    inputs = {
        "batch": original,
        "frames": list(original),
        "single_batches": list(original.split(1)),
        "chunks": [original[:2], original[2:6], original[6:]],
    }
    result = trim_frame_sequence_ends(inputs[representation], start, end)
    assert torch.equal(result, original[start:10 - end])
    assert result.dtype == original.dtype
    assert result.device == original.device
    assert torch.equal(original, before)
    if representation == "batch":
        assert result.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
