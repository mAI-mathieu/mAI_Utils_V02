import pytest
import torch

from nodes.h3_to_ltx_frame_adapter import MAIH3ToLTXFrameAdapter
from utils.h3_to_ltx_frame_adapter import get_ltx_frame_indices, select_ltx_frames


EXAMPLES = [
    (9, 9), (17, 17), (25, 25), (73, 73), (89, 89), (90, 89),
    (105, 105), (107, 105), (121, 121), (124, 121), (137, 137), (209, 209),
]


@pytest.mark.parametrize("source,target", EXAMPLES)
def test_requested_counts_and_endpoints(source, target):
    indices = get_ltx_frame_indices(source)
    assert len(indices) == target
    assert target % 8 == 1
    assert target <= source
    assert indices[0] == 0
    assert indices[-1] == source - 1
    assert len(set(indices)) == target
    assert indices == sorted(indices)


def test_sampling_is_even_across_a_range_of_counts():
    for source in range(9, 513):
        indices = get_ltx_frame_indices(source)
        target = len(indices)
        assert target <= source < target + 8
        assert target % 8 == 1
        assert indices[0] == 0 and indices[-1] == source - 1
        assert all(a < b for a, b in zip(indices, indices[1:]))
        assert all(
            abs(index - i * (source - 1) / (target - 1)) <= 0.5
            for i, index in enumerate(indices)
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("channels", [3, 4])
def test_exact_pixel_selection_and_distributed_removal(dtype, channels):
    images = torch.arange(124, dtype=dtype)[:, None, None, None].expand(124, 2, 3, channels)
    before = images.clone()
    result = select_ltx_frames(images)
    indices = get_ltx_frame_indices(124)
    assert result.shape == (121, 2, 3, channels)
    assert result.dtype == images.dtype and result.device == images.device
    assert torch.equal(result, images[indices])
    assert torch.equal(result[0], images[0])
    assert torch.equal(result[-1], images[-1])
    assert torch.equal(images, before)
    removed = sorted(set(range(124)) - set(indices))
    assert len(removed) == 3
    assert 0 < removed[0] < 41 < removed[1] < 82 < removed[2] < 123


def test_special_pixel_values_are_preserved_bit_for_bit():
    images = torch.randn(124, 2, 3, 4)
    images[0, 0, 0] = torch.tensor([float("nan"), float("inf"), -0.0, -3.0])
    images[-1, 0, 0] = torch.tensor([-float("inf"), float("nan"), 2.0, 0.0])
    before = images.clone()
    result = select_ltx_frames(images)
    expected = images[get_ltx_frame_indices(124)]
    assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(images.view(torch.uint8), before.view(torch.uint8))


@pytest.mark.parametrize("source,target", EXAMPLES)
def test_node_counts_and_compatible_batch_identity(source, target):
    images = torch.rand(source, 2, 3, 3)
    result, source_count, target_count, removed = MAIH3ToLTXFrameAdapter().adapt_frames([images])
    assert (source_count, target_count, removed) == (source, target, source - target)
    assert torch.equal(result, images[get_ltx_frame_indices(source)])
    if source == target:
        assert result is images


@pytest.mark.parametrize("one_frame_batches", [False, True])
@pytest.mark.parametrize("source", [121, 124])
def test_list_element_identity_and_comfyui_batch_output(one_frame_batches, source):
    images = torch.arange(source * 24, dtype=torch.float64).reshape(source, 2, 3, 4)
    frames = list(images.split(1)) if one_frame_batches else list(images)
    selected = select_ltx_frames(frames)
    indices = get_ltx_frame_indices(source)
    assert all(frame is frames[index] for frame, index in zip(selected, indices))
    assert selected[0] is frames[0] and selected[-1] is frames[-1]
    if source == 121:
        assert selected is frames
    for input_value in (frames, [frames]):
        result, count, target, removed = MAIH3ToLTXFrameAdapter().adapt_frames(input_value)
        assert torch.equal(result, images[indices])
        assert (count, target, removed) == (source, len(indices), source - len(indices))


@pytest.mark.parametrize("count", range(9))
def test_empty_and_short_inputs(count):
    message = "Input image batch contains no frames." if count == 0 else (
        "At least 9 input frames are required for LTX-compatible video refinement."
    )
    for input_value in (torch.zeros(count, 2, 3, 3), [torch.zeros(2, 3, 3) for _ in range(count)]):
        with pytest.raises(ValueError, match=message):
            MAIH3ToLTXFrameAdapter().adapt_frames(input_value)


@pytest.mark.parametrize("count", [-1, True, 9.5, "9", None])
def test_invalid_counts(count):
    with pytest.raises(ValueError, match="non-negative integer"):
        get_ltx_frame_indices(count)


@pytest.mark.parametrize(
    "images,message",
    [
        (None, "IMAGE batch tensor"),
        (torch.zeros(9, 2, 3), "images must have shape"),
        (torch.zeros(9, 0, 3, 3), "non-empty"),
        ([None] * 9, "HWC or one-frame"),
        ([torch.zeros(2, 2, 3, 3)] * 9, "exactly one frame"),
        ([torch.zeros(0, 3, 3)] * 9, "empty dimensions"),
        ([torch.zeros(2, 3, 3)] * 8 + [torch.zeros(3, 3, 3)], "same height"),
        ([torch.zeros(2, 3, 3)] * 8 + [torch.zeros(2, 3, 4)], "same height"),
        ([torch.zeros(2, 3, 3)] * 8 + [torch.zeros(2, 3, 3, dtype=torch.float64)], "dtype and device"),
    ],
)
def test_invalid_image_inputs(images, message):
    with pytest.raises(ValueError, match=message):
        select_ltx_frames(images)


def test_node_contract_and_logging(capsys):
    node = MAIH3ToLTXFrameAdapter()
    assert node.INPUT_TYPES() == {"required": {"images": ("IMAGE",)}}
    assert node.INPUT_IS_LIST is True
    assert node.RETURN_TYPES == ("IMAGE", "INT", "INT", "INT")
    assert node.RETURN_NAMES == ("images", "source_frames", "target_frames", "removed_frames")
    assert node.CATEGORY == "mAI / Image"
    assert node.FUNCTION == "adapt_frames"
    node.adapt_frames(torch.zeros(124, 1, 1, 3))
    assert capsys.readouterr().out == "[mAI H3 to LTX Frame Adapter] 124 → 121 frames, removed 3\n"
    node.adapt_frames(torch.zeros(121, 1, 1, 3))
    assert capsys.readouterr().out == "[mAI H3 to LTX Frame Adapter] 121 frames already LTX-compatible\n"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_device_is_preserved():
    images = torch.rand(124, 2, 3, 4, device="cuda", dtype=torch.float16)
    result = select_ltx_frames(images)
    assert result.device == images.device and result.dtype == images.dtype
    assert torch.equal(result, images[get_ltx_frame_indices(124)])
    with pytest.raises(ValueError, match="dtype and device"):
        select_ltx_frames([images[0].cpu()] + list(images[1:]))
