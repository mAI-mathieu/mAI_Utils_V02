from fractions import Fraction
import json
import wave

import pytest

from utils.h3_reference_decode import decode_settings, fit_video_dimensions


@pytest.mark.parametrize("seconds,side", [(0, 768), (16, 768), (float("nan"), 768), (True, 768), ("15", 768), (15, 0), (15, 2049), (15, True), (15, 128.5)])
def test_invalid_decode_settings(seconds, side):
    with pytest.raises(ValueError):
        decode_settings(seconds, side)


def test_video_size_preserves_aspect_and_never_upscales():
    assert fit_video_dimensions(1920, 1080, 768) == (768, 432)
    assert fit_video_dimensions(32, 16, 768) == (32, 16)


def dependencies():
    return pytest.importorskip("av"), pytest.importorskip("numpy"), pytest.importorskip("torch")


def write_video(path, rate=24, seconds=2, offset=0):
    av, np, _ = dependencies()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        for index in range(round(rate * seconds)):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), index * 2, dtype=np.uint8), format="rgb24")
            frame.pts = index + offset
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("rate", [12, 24, 30, 60])
def test_real_video_is_resampled_to_24fps_and_aligned(tmp_path, rate):
    _, _, torch = dependencies()
    from utils.h3_reference_decode import load_reference_video

    path = tmp_path / "clip.mp4"
    write_video(path, rate)
    frames = load_reference_video(path, 15, 64)
    assert frames.shape == (39, 32, 64, 3)  # 48 samples, trimmed to 17*2+5.
    assert frames.dtype == torch.float32
    assert 0 <= frames.min() <= frames.max() <= 1
    # At 1s the sample should be the source frame at (or just before) 1s.
    assert abs(frames[24].mean().item() * 255 - rate * 2) < 6
    assert load_reference_video(path, 1, 64).shape[0] == 22


def test_video_timestamps_can_start_after_zero(tmp_path):
    dependencies()
    from utils.h3_reference_decode import load_reference_video

    path = tmp_path / "offset.mp4"
    write_video(path, 24, offset=48)
    assert load_reference_video(path, 15, 64).shape[0] == 39


def test_too_short_video_is_rejected(tmp_path):
    dependencies()
    from utils.h3_reference_decode import load_reference_video

    path = tmp_path / "short.mp4"
    write_video(path, 24, seconds=0.1)
    with pytest.raises(ValueError, match="at least 5 frames"):
        load_reference_video(path, 15, 64)


def test_real_audio_is_bounded_float_stereo_48khz(tmp_path):
    _, np, torch = dependencies()
    from utils.h3_reference_decode import load_reference_audio

    path = tmp_path / "tone.wav"
    samples = (np.sin(np.arange(32000) * 0.1) * 20000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())
    audio = load_reference_audio(path, 0.5)
    assert audio["sample_rate"] == 48000
    assert audio["waveform"].shape == (1, 2, 24000)
    assert audio["waveform"].dtype == torch.float32
    torch.testing.assert_close(audio["waveform"][:, 0], audio["waveform"][:, 1])


def test_missing_video_audio_is_optional(tmp_path):
    dependencies()
    from utils.h3_reference_decode import load_reference_audio

    path = tmp_path / "silent.mp4"
    write_video(path)
    assert load_reference_audio(path, 1, optional=True) is None
    with pytest.raises(ValueError, match="no audio stream"):
        load_reference_audio(path, 1)


def test_image_rgb_exif_orientation(tmp_path):
    dependencies()
    Image = pytest.importorskip("PIL.Image")
    from utils.h3_reference_decode import load_reference_image

    path = tmp_path / "rotated.png"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGBA", (8, 4), (255, 0, 0, 255)).save(path, exif=exif)
    image = load_reference_image(path)
    assert image.shape == (1, 8, 4, 3)
    assert image[0, 0, 0].tolist() == [1, 0, 0]


def test_slot_order_soundtrack_pairing_and_unused_outputs(monkeypatch):
    from types import SimpleNamespace
    import utils.h3_reference_decode as module

    def video(path, *args):
        return SimpleNamespace(shape=(22, 32, 64, 3), name=path)

    monkeypatch.setattr(module, "load_reference_video", video)
    monkeypatch.setattr(module, "load_reference_image", lambda path: path)
    calls = []

    def audio(path, seconds, **kwargs):
        calls.append((path, seconds))
        return None if path == "silent" else path + " audio"

    monkeypatch.setattr(module, "load_reference_audio", audio)
    values = [("audio", "voice"), ("video", "silent"), ("image", "second"), ("video", "movie"), ("image", "first")]
    refs = [SimpleNamespace(kind=kind, path=path) for kind, path in values]
    entries = [{"kind": kind, "name": path} for kind, path in values]
    slots, report = module.load_reference_slots(refs, entries)
    assert len(slots) == 18
    assert slots[:3] == ("second", "first", None)
    assert slots[9].name == "silent" and slots[10].name == "movie"
    assert slots[12:15] == (None, "movie audio", None)
    assert slots[15:] == ("voice audio", None, None)
    assert report[0]["label"] == "<Audio 2>"
    assert report[3]["soundtrack_label"] == "<Audio 1>"
    assert ("movie", 22 / 24) in calls
    slots, _ = module.load_reference_slots(refs, entries, include_video_audio=False)
    assert slots[12:15] == (None, None, None)


def test_invalid_media_error_names_the_reference(monkeypatch):
    from types import SimpleNamespace
    import utils.h3_reference_decode as module

    def fail(path):
        raise OSError("invalid image")

    monkeypatch.setattr(module, "load_reference_image", fail)
    with pytest.raises(ValueError, match="my picture.png"):
        module.load_reference_slots([SimpleNamespace(kind="image", path="bad")], [{"kind": "image", "name": "my picture.png"}])
