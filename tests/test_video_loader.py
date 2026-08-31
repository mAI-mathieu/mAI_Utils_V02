import sys
from fractions import Fraction
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nodes.video_loader import MAIVideoLoader
from utils.video_loader import get_frame_batch_info


def test_get_frame_batch_info_returns_count_width_and_height():
    frames = torch.zeros(12, 720, 1280, 3)

    assert get_frame_batch_info(frames) == (12, 1280, 720)


@pytest.mark.parametrize(
    "frames",
    [
        torch.zeros(720, 1280, 3),
        torch.zeros(0, 720, 1280, 3),
        torch.zeros(1, 0, 1280, 3),
    ],
)
def test_get_frame_batch_info_rejects_invalid_batches(frames):
    with pytest.raises(ValueError):
        get_frame_batch_info(frames)


def test_node_declares_stable_output_contract():
    assert MAIVideoLoader.RETURN_TYPES == (
        "IMAGE",
        "FLOAT",
        "AUDIO",
        "INT",
        "INT",
        "INT",
    )
    assert MAIVideoLoader.RETURN_NAMES == (
        "frames",
        "fps",
        "audio",
        "frame_count",
        "width",
        "height",
    )


def test_node_loads_native_comfyui_video_components(monkeypatch):
    frames = torch.rand(7, 360, 640, 3)
    audio = {"waveform": torch.rand(1, 2, 48000), "sample_rate": 48000}
    components = SimpleNamespace(
        images=frames,
        frame_rate=Fraction(30000, 1001),
        audio=audio,
    )

    folder_paths = ModuleType("folder_paths")
    folder_paths.get_annotated_filepath = lambda video: f"input/{video}"

    class VideoFromFile:
        def __init__(self, path):
            assert path == "input/clip.mp4"

        def get_components(self):
            return components

    comfy_api = ModuleType("comfy_api")
    comfy_api_latest = ModuleType("comfy_api.latest")
    comfy_api_latest.InputImpl = SimpleNamespace(VideoFromFile=VideoFromFile)
    comfy_api.latest = comfy_api_latest

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_api_latest)

    result = MAIVideoLoader().load_video("clip.mp4")

    assert result[0] is frames
    assert result[1] == pytest.approx(30000 / 1001)
    assert result[2] is audio
    assert result[3:] == (7, 640, 360)


def test_node_preserves_missing_audio(monkeypatch):
    components = SimpleNamespace(
        images=torch.rand(1, 32, 48, 3),
        frame_rate=Fraction(24, 1),
        audio=None,
    )
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_annotated_filepath = lambda video: video

    class VideoFromFile:
        def __init__(self, path):
            pass

        def get_components(self):
            return components

    comfy_api = ModuleType("comfy_api")
    comfy_api_latest = ModuleType("comfy_api.latest")
    comfy_api_latest.InputImpl = SimpleNamespace(VideoFromFile=VideoFromFile)
    comfy_api.latest = comfy_api_latest

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_api_latest)

    result = MAIVideoLoader().load_video("silent.mp4")

    assert result[2] is None
