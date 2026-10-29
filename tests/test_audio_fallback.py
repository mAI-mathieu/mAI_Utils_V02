import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from nodes.audio_fallback import MAIAudioFallback
from utils.audio_fallback import resolve_audio_fallback, silence_sample_count


@pytest.mark.parametrize("source", [
    None, {}, {"waveform": None},
    {"waveform": torch.empty(1, 1, 0), "sample_rate": 44100},
])
def test_missing_or_empty_audio_produces_stereo_silence(source):
    result = resolve_audio_fallback(source, 12, 24)
    assert result["sample_rate"] == 48000
    waveform = result["waveform"]
    assert waveform.shape == (1, 2, 24000)
    assert waveform.dtype == torch.float32
    assert waveform.device.type == "cpu"
    assert torch.count_nonzero(waveform).item() == 0


@pytest.mark.parametrize("silent", [True, False])
def test_populated_audio_passes_through_without_duration_or_metadata_changes(silent):
    waveform = torch.zeros(2, 3, 100, dtype=torch.float64)
    if not silent:
        waveform[..., 50] = 0.75
    original = waveform.clone()
    source = {"waveform": waveform, "sample_rate": 44100, "metadata": {"test": True}}
    result = MAIAudioFallback().run(frame_count=120, fps=24, audio=source)
    assert len(result) == 1
    assert result[0] is source
    assert result[0]["waveform"] is waveform
    torch.testing.assert_close(waveform, original, rtol=0, atol=0)


@pytest.mark.parametrize("frames, fps", [(81, 23.976), (121, 29.97), (1, 100000)])
def test_fallback_duration_rounds_to_nearest_sample(frames, fps):
    count = silence_sample_count(frames, fps)
    assert count == max(1, round(frames / fps * 48000))
    assert resolve_audio_fallback(None, frames, fps)["waveform"].shape[-1] == count


@pytest.mark.parametrize("frames", [0, -1, 1.5, True, "120"])
def test_invalid_frame_count_has_clear_error(frames):
    with pytest.raises(ValueError, match="frame_count"):
        resolve_audio_fallback(None, frames, 24)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, "24"])
def test_invalid_fps_has_clear_error(fps):
    with pytest.raises(ValueError, match="fps"):
        resolve_audio_fallback(None, 120, fps)


@pytest.mark.parametrize("source, message", [
    ([], "dictionary"),
    ({"waveform": []}, "waveform"),
    ({"waveform": torch.zeros(2, 10)}, "waveform"),
    ({"waveform": torch.zeros(0, 2, 10)}, "waveform"),
    ({"waveform": torch.zeros(1, 0, 10)}, "waveform"),
    ({"waveform": torch.zeros(1, 2, 10, dtype=torch.int16)}, "waveform"),
    ({"waveform": torch.zeros(1, 2, 10)}, "sample_rate"),
    ({"waveform": torch.zeros(1, 2, 10), "sample_rate": 0}, "sample_rate"),
    ({"waveform": torch.zeros(1, 2, 10), "sample_rate": True}, "sample_rate"),
    ({"waveform": torch.zeros(1, 2, 10), "sample_rate": 48000.0}, "sample_rate"),
])
def test_malformed_populated_audio_has_clear_error(source, message):
    with pytest.raises(ValueError, match=message):
        resolve_audio_fallback(source, 120, 24)


def test_node_defaults_and_socket_contract():
    node = MAIAudioFallback()
    assert node.CATEGORY == "mAI / Utils"
    assert node.RETURN_TYPES == ("AUDIO",)
    assert node.RETURN_NAMES == ("audio",)
    inputs = node.INPUT_TYPES()
    assert inputs["optional"]["audio"] == ("AUDIO",)
    defaults = {name: spec[1]["default"] for name, spec in inputs["required"].items()}
    result = node.run(**defaults)[0]
    assert result["waveform"].shape == (1, 2, 240000)
    assert result["sample_rate"] == 48000


def test_package_registration_preserves_existing_nodes(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    name = "mai_audio_fallback_registration"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIAudioFallback"].__name__ == "MAIAudioFallback"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIAudioFallback"] == "mAI audio fallback"
    assert {"MAIVideoLoader", "MAIGPUVideoCombine", "MAIAutoSeamlessLoop"} <= package.NODE_CLASS_MAPPINGS.keys()
    tree = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
    mapping = next(
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                for target in node.targets)
    )
    assert {key.value for key in mapping.keys} <= package.NODE_CLASS_MAPPINGS.keys()
