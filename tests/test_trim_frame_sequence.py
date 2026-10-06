import shutil
import subprocess
from pathlib import Path

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
    assert list(inputs) == ["frames", "trim_start", "trim_end"]
    for name in ("trim_start", "trim_end"):
        assert inputs[name][0] == "INT"
        assert inputs[name][1]["default"] == 0
        assert inputs[name][1]["min"] == 0


@pytest.mark.parametrize("representation", ["batch", "list", "wrapped_list", "direct"])
def test_node_handles_comfyui_list_inputs_and_wrapped_widgets(representation):
    frames = torch.arange(30).reshape(10, 1, 1, 3)
    chunks = [frames[:3], frames[3:8], frames[8:]]
    inputs = {"batch": [frames], "list": chunks, "wrapped_list": [chunks], "direct": frames}
    result, count = MAITrimFrameSequence().run(inputs[representation], [2], [3])
    assert torch.equal(result, frames[2:7])
    assert count == 5


def test_direct_scalar_settings():
    frames = torch.rand(5, 2, 2, 3)
    result, count = MAITrimFrameSequence().run(frames, 0, 1)
    assert torch.equal(result, frames[:4])
    assert count == 4


@pytest.mark.parametrize("start,end", [([], [0]), ([1, 2], [0]), ([0], []), ([0], [1, 2])])
def test_rejects_ambiguous_settings(start, end):
    with pytest.raises(ValueError, match="one value"):
        MAITrimFrameSequence().run([torch.zeros(5, 2, 2, 3)], start, end)


def test_default_counts_keep_single_frame():
    frames = torch.rand(1, 2, 2, 3)
    result, count = MAITrimFrameSequence().run([frames])
    assert torch.equal(result, frames)
    assert count == 1


def test_excessive_trim_reports_clear_error():
    with pytest.raises(ValueError, match="at least one frame must remain"):
        MAITrimFrameSequence().run([torch.zeros(5, 2, 2, 3)], [2], [3])


def test_frontend_restores_legacy_widget_settings():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for frontend tests")
    script = Path(__file__).with_name("trim_frame_sequence_frontend.mjs")
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
