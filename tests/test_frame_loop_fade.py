import pytest
import torch

from nodes.frame_loop_fade import MAIFrameLoopFade
from utils.frame_sequence import append_loop_fade


@pytest.mark.parametrize("representation", ["batch", "frames", "chunks"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_fade_preserves_source_and_blends_last_into_first(representation, dtype):
    source = torch.tensor([0.0, 0.2, 1.0], dtype=dtype).reshape(3, 1, 1, 1).expand(-1, 2, 2, 3).clone()
    before = source.clone()
    inputs = {"batch": source, "frames": list(source), "chunks": [source[:2], source[2:]]}
    result = append_loop_fade(inputs[representation], 4)
    assert result.shape == (7, 2, 2, 3)
    assert torch.equal(result[:3], before)
    expected = torch.tensor([0.75, 0.5, 0.25, 0.0], dtype=dtype)
    torch.testing.assert_close(result[3:, 0, 0, 0], expected)
    assert torch.equal(result[-1], source[0])
    assert result.dtype == source.dtype
    assert result.device == source.device
    assert torch.equal(source, before)


@pytest.mark.parametrize("count", [0, 1, 5])
def test_single_image_and_small_fades(count):
    source = torch.rand(1, 2, 3, 4)
    result = append_loop_fade(source, count)
    assert torch.equal(result, source.expand(count + 1, -1, -1, -1))
    if count == 0:
        assert result is source


def test_one_fade_frame_appends_first_image_exactly():
    source = torch.rand(3, 2, 2, 3)
    assert torch.equal(append_loop_fade(source, 1), torch.cat([source, source[:1]]))


def test_zero_fade_combines_list_in_order():
    source = torch.rand(5, 2, 2, 3)
    assert torch.equal(append_loop_fade([source[:2], source[2:]], 0), source)


@pytest.mark.parametrize("count", [-1, 1.5, "4", True, None])
def test_invalid_fade_counts(count):
    with pytest.raises(ValueError, match="non-negative integer"):
        append_loop_fade(torch.zeros(2, 1, 1, 3), count)


@pytest.mark.parametrize(
    "images,message",
    [
        ([], "at least one"),
        (None, "IMAGE tensor"),
        (torch.zeros(0, 2, 2, 3), "empty dimensions"),
        (torch.zeros(2, 2), "IMAGE tensor"),
        (torch.zeros(2, 2, 2, 3, dtype=torch.int64), "floating-point"),
        ([torch.zeros(1, 2, 2, 3), torch.zeros(1, 3, 2, 3)], "same height"),
        ([torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 4)], "same height"),
        ([torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 3, dtype=torch.float64)], "dtype and device"),
    ],
)
def test_invalid_image_inputs(images, message):
    with pytest.raises(ValueError, match=message):
        append_loop_fade(images, 4)


@pytest.mark.parametrize("representation", ["batch", "list", "wrapped_list", "direct"])
def test_node_processes_comfyui_sequence_as_one_fade(representation):
    source = torch.rand(5, 2, 2, 3)
    chunks = [source[:2], source[2:]]
    inputs = {"batch": [source], "list": chunks, "wrapped_list": [chunks], "direct": source}
    result, count = MAIFrameLoopFade().run(inputs[representation], [4])
    assert count == 9
    assert torch.equal(result[:5], source)
    assert torch.equal(result[-1], source[0])


def test_node_default_fade_and_socket_contract():
    node = MAIFrameLoopFade()
    source = torch.rand(2, 2, 2, 3)
    result, count = node.run([source])
    assert count == result.shape[0] == 26
    assert node.RETURN_TYPES == ("IMAGE", "INT")
    assert node.RETURN_NAMES == ("images", "frame_count")
    assert node.CATEGORY == "mAI / Image"
    assert node.INPUT_IS_LIST is True
    assert node.INPUT_TYPES()["required"]["fade_frames"][1]["default"] == 24


@pytest.mark.parametrize("count", [[], [1, 2]])
def test_node_rejects_multiple_or_missing_fade_settings(count):
    with pytest.raises(ValueError, match="one value"):
        MAIFrameLoopFade().run([torch.rand(2, 2, 2, 3)], count)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fade_stays_on_cuda():
    source = torch.rand(3, 2, 2, 3, device="cuda")
    result = append_loop_fade(source, 4)
    assert result.device == source.device
    assert torch.equal(result[:3], source)
    assert torch.equal(result[-1], source[0])
