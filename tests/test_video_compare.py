import ast
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from nodes.video_compare import MAIVideoCompare
from utils.video_compare import (
    compare_batches, compare_settings, prepare_compare_pixels, preview_dimensions,
    resolve_compare_device, write_compare_frames,
)
import utils.video_compare as compare_module


@pytest.mark.parametrize("representation", ["batch", "chunks", "frames", "wrapped"])
def test_sequence_order_and_shorter_duration(tmp_path, monkeypatch, representation):
    a = torch.stack([torch.full((2, 4, 3), value) for value in (0.0, 0.5, 1.0)])
    b = torch.ones(2, 4, 2, 4)
    values = {"batch": [a], "chunks": [a[:1], a[1:]], "frames": list(a), "wrapped": [[a[:1], a[1:]]]}
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_temp_directory=lambda: str(tmp_path)))
    output = MAIVideoCompare().run(values[representation], [b], [12.5], [64])
    assert output["result"] == ()
    data, = output["ui"]["video_compare"]
    assert data["frame_count"] == 2
    assert data["source_counts"] == [3, 2]
    assert data["fps"] == 12.5
    assert (data["width"], data["height"]) == (4, 2)
    folder = tmp_path / data["subfolder"]
    assert sorted(path.name for path in folder.iterdir()) == ["a_000000.png", "a_000001.png", "b_000000.png", "b_000001.png"]
    with Image.open(folder / "a_000001.png") as image:
        assert image.getpixel((0, 0)) == (128, 128, 128)
    with Image.open(folder / "b_000000.png") as image:
        assert image.mode == "RGBA" and image.size == (2, 4)
    assert a[1, 0, 0, 0] == 0.5  # Preview conversion never mutates source pixels.


@pytest.mark.parametrize("channels,mode", [(1, "L"), (3, "RGB"), (4, "RGBA")])
def test_resize_channels_and_sanitized_pixels(tmp_path, channels, mode):
    frames = torch.ones(1, 80, 160, channels, dtype=torch.bfloat16)
    frames[:, 0, 0] = float("nan")
    frames[:, 0, 1] = float("inf")
    progress = []
    write_compare_frames([frames], tmp_path, "a", 1, 64, on_frame=lambda: progress.append(1))
    with Image.open(tmp_path / "a_000000.png") as image:
        assert image.size == (64, 32) and image.mode == mode
    assert progress == [1]
    assert torch.isnan(frames[0, 0, 0]).all()


def test_preview_dimensions_never_upscale_or_become_empty():
    assert preview_dimensions(1920, 1080, 768) == (768, 432)
    assert preview_dimensions(30, 40, 768) == (30, 40)
    assert preview_dimensions(100000, 1, 64) == (64, 1)


@pytest.mark.parametrize("fps,size", [(0, 768), (float("nan"), 768), (float("inf"), 768), (True, 768), (121, 768), (24, 0), (24, 4097), (24, 128.5), (24, True)])
def test_invalid_settings(fps, size):
    with pytest.raises(ValueError):
        compare_settings(fps, size)


@pytest.mark.parametrize("frames", [[], None, torch.empty(0, 2, 2, 3), torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 3, dtype=torch.uint8), [torch.zeros(1, 2, 2, 3), torch.zeros(1, 3, 2, 3)]])
def test_invalid_sequences(frames):
    with pytest.raises(ValueError):
        compare_batches(frames, "frames_a")


def test_cancellation_removes_partial_files_and_keeps_previous_previews(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_temp_directory=lambda: str(tmp_path)))
    node = MAIVideoCompare()
    frames = torch.rand(2, 2, 2, 3)
    first = node.run(frames, frames)["ui"]["video_compare"][0]
    previous = tmp_path / first["subfolder"]
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("cancelled")

    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(throw_exception_if_processing_interrupted=interrupt))
    with pytest.raises(RuntimeError, match="cancelled"):
        node.run(frames, frames)
    assert list(tmp_path.iterdir()) == [previous]
    assert len(list(previous.iterdir())) == 4


@pytest.mark.parametrize("name", ["fps", "preview_max_size"])
def test_rejects_ambiguous_widget_lists(name):
    frames = torch.zeros(1, 2, 2, 3)
    with pytest.raises(ValueError, match="one value"):
        # folder_paths is imported before settings, matching the real runtime.
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(sys.modules, "folder_paths", SimpleNamespace())
            MAIVideoCompare().run(frames, frames, **{name: [1, 2]})


def test_registration_preserves_all_existing_nodes(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    name = "mai_video_compare_registration"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIVideoCompare"].__name__ == "MAIVideoCompare"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIVideoCompare"] == "mAI video compare"
    assert MAIVideoCompare.CATEGORY == "mAI / Image"
    assert MAIVideoCompare.OUTPUT_NODE and MAIVideoCompare.INPUT_IS_LIST
    assert MAIVideoCompare.RETURN_TYPES == ()
    assert package.WEB_DIRECTORY == "./web"
    tree = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
    mapping = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS" for target in node.targets))
    assert {key.value for key in mapping.keys} <= package.NODE_CLASS_MAPPINGS.keys()
    assert {"MAIVideoLoader", "MAIGPUVideoCombine", "MAITrimFrameSequence"} <= package.NODE_CLASS_MAPPINGS.keys()


def test_frontend_player():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run([node, str(Path(__file__).with_name("video_compare_frontend.mjs"))], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_mode_does_not_query_cuda(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(get_torch_device=lambda: "cpu"))

    def forbidden():
        raise AssertionError("CPU mode queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert resolve_compare_device("cpu") == torch.device("cpu")
    write_compare_frames([torch.ones(1, 80, 160, 3)], tmp_path, "a", 1, 64)
    assert (tmp_path / "a_000000.png").is_file()


def test_cpu_inputs_use_comfyui_selected_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(get_torch_device=lambda: "cuda:2"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_compare_device("cpu") == torch.device("cuda:2")


def test_unavailable_cuda_uses_cpu(monkeypatch):
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_compare_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pixel_conversion_stays_on_compute_device_without_mutation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = torch.tensor([[[float("nan"), float("inf"), -1.0], [0.0, 0.5, 1.0]]])
    result = prepare_compare_pixels(source, device)
    assert result.device.type == device
    assert result.dtype == torch.uint8
    assert result.cpu().tolist() == [[[0, 255, 0], [0, 128, 255]]]
    assert torch.isnan(source[0, 0, 0]) and source[0, 1, 1] == 0.5


@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_real_cuda_resize_and_png_match_cpu(tmp_path, monkeypatch, source_device, channels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Both a CPU loader and an upstream GPU tensor must use CUDA preparation.
    source = torch.rand(2, 80, 160, channels, device=source_device)
    original = source.clone()
    seen = []
    prepare = compare_module.prepare_compare_pixels
    resize = compare_module.resize_separable

    def observed_resize(pixels, tables):
        seen.append(("resize", pixels.device.type))
        return resize(pixels, tables)

    def observed_prepare(*args):
        result = prepare(*args)
        seen.append(("quantize", result.device.type))
        assert result.dtype == torch.uint8
        return result

    monkeypatch.setattr(compare_module, "resize_separable", observed_resize)
    monkeypatch.setattr(compare_module, "prepare_compare_pixels", observed_prepare)
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(get_torch_device=lambda: "cuda:0"))
    write_compare_frames([source], tmp_path, "gpu", 2, 64)
    assert seen == [("resize", "cuda"), ("quantize", "cuda")] * 2
    write_compare_frames([source.cpu()], tmp_path, "cpu", 2, 64, compute_device="cpu")
    for index in range(2):
        with Image.open(tmp_path / f"gpu_{index:06d}.png") as gpu, Image.open(tmp_path / f"cpu_{index:06d}.png") as cpu:
            assert gpu.size == cpu.size == (64, 32)
            gpu_bytes = torch.frombuffer(bytearray(gpu.tobytes()), dtype=torch.uint8).int()
            cpu_bytes = torch.frombuffer(bytearray(cpu.tobytes()), dtype=torch.uint8).int()
            differences = gpu_bytes - cpu_bytes
            assert differences.abs().max() <= 1
    torch.testing.assert_close(source, original)


def test_cuda_errors_are_reported_and_partial_previews_removed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_temp_directory=lambda: str(tmp_path)))
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(get_torch_device=lambda: "cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    attempts = []

    def fail(frame, device, tables):
        attempts.append(device.type)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(compare_module, "prepare_compare_pixels", fail)
    frames = torch.zeros(1, 4, 4, 3)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA out of memory"):
        MAIVideoCompare().run(frames, frames)
    assert attempts == ["cuda"]
    assert list(tmp_path.iterdir()) == []
