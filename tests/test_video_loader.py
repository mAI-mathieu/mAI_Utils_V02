import sys
from fractions import Fraction
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nodes.video_loader import MAIVideoLoader
from utils.video_loader import get_frame_batch_info, load_video_components


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


@pytest.mark.parametrize(
    ("limit_args", "expected_count"),
    [
        ({}, 105),
        ({"frame_limit": -1}, 105),
        ({"frame_limit": 100}, 100),
        ({"frame_limit": 200}, 105),
    ],
)
def test_node_loads_native_comfyui_video_components(monkeypatch, limit_args, expected_count):
    frames = torch.rand(105, 12, 16, 3)
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

    result = MAIVideoLoader().load_video("clip.mp4", **limit_args)

    assert torch.equal(result[0], frames[:expected_count])
    assert result[1] == pytest.approx(30000 / 1001)
    assert result[2] is audio
    assert result[3:] == (expected_count, 16, 12)


@pytest.mark.parametrize("with_audio", [False, True])
def test_frame_limit_skips_later_video_packets_and_preserves_audio(with_audio):
    video = SimpleNamespace(type="video")
    audio = SimpleNamespace(type="audio")
    demuxed = []
    decoded = []

    def packet(stream, *values):
        def decode():
            decoded.extend(values)
            return values

        return SimpleNamespace(stream=stream, decode=decode)

    class Container:
        def demux(self, *streams):
            for item in [packet(video, 1, 2, 3), packet(video, 4), packet(audio, 5)]:
                if item.stream in streams:
                    demuxed.append(item.stream.type)
                    yield item

    class VideoFromFile:
        def __init__(self, path):
            pass

        def _decode_frames(self, container, video_stream, audio_stream, audio_frames, alphas):
            streams = (video_stream, audio_stream) if audio_stream else (video_stream,)
            for item in container.demux(*streams):
                if item.stream is video_stream:
                    yield from item.decode()
                else:
                    audio_frames.extend(item.decode())

        def get_components(self):
            audio_frames = []
            images = list(self._decode_frames(Container(), video, audio if with_audio else None, audio_frames, []))
            return SimpleNamespace(images=images, audio=audio_frames)

    result = load_video_components(VideoFromFile, "clip.mp4", frame_limit=2)

    assert result.images == [1, 2]
    assert 4 not in decoded
    if with_audio:
        assert result.audio == [5]
    else:
        assert demuxed == ["video"]


@pytest.mark.parametrize("frame_limit", [-2, 0])
def test_node_rejects_invalid_frame_limits(frame_limit):
    with pytest.raises(ValueError, match="-1 .* or a positive integer"):
        MAIVideoLoader().load_video("clip.mp4", frame_limit)


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
