import pytest
import torch

from nodes.trim_frame_sequence import MAITrimFrameSequence


def test_node_contract_and_defaults():
    assert MAITrimFrameSequence.INPUT_IS_LIST is True
    assert MAITrimFrameSequence.RETURN_TYPES == ("IMAGE", "INT")
    assert MAITrimFrameSequence.RETURN_NAMES == ("frames", "frame_count")
    assert MAITrimFrameSequence.CATEGORY == "mAI / Image"
    inputs = MAITrimFrameSequence.INPUT_TYPES()["required"]
    assert inputs["frames"] == ("IMAGE",)
    assert inputs["trim_mode"][1]["default"] == "start"
    assert inputs["trim_amount"][1]["default"] == 0


@pytest.mark.parametrize("representation", ["batch", "list", "wrapped_list", "direct"])
def test_node_handles_comfyui_list_inputs_and_wrapped_widgets(representation):
    frames = torch.arange(30).reshape(10, 1, 1, 3)
    chunks = [frames[:3], frames[3:8], frames[8:]]
    inputs = {"batch": [frames], "list": chunks, "wrapped_list": [chunks], "direct": frames}
    result, count = MAITrimFrameSequence().run(inputs[representation], ["both ends"], [2])
    assert torch.equal(result, frames[2:8])
    assert count == 6


def test_direct_scalar_settings():
    frames = torch.rand(5, 2, 2, 3)
    result, count = MAITrimFrameSequence().run(frames, "end", 1)
    assert torch.equal(result, frames[:4])
    assert count == 4


@pytest.mark.parametrize("mode,amount", [([], [0]), (["start", "end"], [0]), (["start"], []), (["start"], [1, 2])])
def test_rejects_ambiguous_settings(mode, amount):
    with pytest.raises(ValueError, match="one value"):
        MAITrimFrameSequence().run([torch.zeros(5, 2, 2, 3)], mode, amount)
