import sys
from types import SimpleNamespace

import pytest
import torch

from nodes.prepare_image_for_minimax_h3 import MAIPrepareImageForMinimaxH3
from utils import minimax_h3_resize as resize
from utils.minimax_h3 import RESIZE_ALGORITHM_OPTIONS


def select_device(monkeypatch, device):
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(
        get_torch_device=lambda: torch.device(device),
    ))


def test_prefers_comfyui_selected_gpu_for_cpu_input(monkeypatch):
    select_device(monkeypatch, "cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resize.resolve_resize_device(torch.device("cpu")) == torch.device("cuda:1")


def test_cuda_unavailable_selects_cpu(monkeypatch):
    select_device(monkeypatch, "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resize.resolve_resize_device(torch.device("cpu")) == torch.device("cpu")


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_cpu_mode_and_other_accelerators_do_not_query_cuda(monkeypatch, device):
    select_device(monkeypatch, device)

    def forbidden():
        raise AssertionError("CPU mode must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    result = resize.resize_minimax_image(torch.full((2, 5, 9, 3), 0.3), 13, 7, "lanczos")
    assert result.device.type == "cpu"
    torch.testing.assert_close(result, torch.full((2, 7, 13, 3), 0.3))


@pytest.mark.parametrize("algorithm", RESIZE_ALGORITHM_OPTIONS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_cpu_resize_preserves_batch_dtype_channels_and_input(monkeypatch, algorithm, dtype, channels):
    select_device(monkeypatch, "cpu")
    image = torch.stack([
        torch.full((5, 9, channels), 0.2, dtype=dtype),
        torch.full((5, 9, channels), 0.7, dtype=dtype),
    ])
    original = image.clone()
    output = resize.resize_minimax_image(image, 13, 7, algorithm)
    assert output.dtype == dtype and output.device.type == "cpu"
    torch.testing.assert_close(output, image[:, :1, :1].expand(2, 7, 13, channels))
    assert torch.equal(image, original)


@pytest.mark.parametrize("message", [
    "CUDA out of memory", "CUDA error: no kernel image is available",
    "CUDA driver error", "cuDNN error", "CUBLAS_STATUS_EXECUTION_FAILED",
])
def test_gpu_failures_retry_once_on_cpu(monkeypatch, message):
    select_device(monkeypatch, "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original_resize = resize.resize_image_batch
    devices = []

    def failing_gpu(*args, **kwargs):
        device = kwargs["compute_device"].type
        devices.append(device)
        if device == "cuda":
            raise RuntimeError(message)
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(resize, "resize_image_batch", failing_gpu)
    image = torch.full((2, 5, 9, 4), 0.3)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        result = resize.resize_minimax_image(image, 13, 7, "lanczos")
    assert devices == ["cuda", "cpu"]
    assert result.device.type == "cpu" and result.dtype == image.dtype
    torch.testing.assert_close(result, torch.full((2, 7, 13, 4), 0.3))


def test_device_initialization_failure_retries_on_cpu(monkeypatch):
    def fail(device):
        raise RuntimeError("CUDA initialization error")

    monkeypatch.setattr(resize, "resolve_resize_device", fail)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        result = resize.resize_minimax_image(torch.ones(1, 5, 9, 1), 13, 7, "bilinear")
    assert result.device.type == "cpu" and result.shape == (1, 7, 13, 1)


@pytest.mark.parametrize("device,message", [
    ("cpu", "CPU allocation failed"), ("cuda:0", "unexpected shape error"),
])
def test_unrelated_and_cpu_errors_remain_visible(monkeypatch, device, message):
    select_device(monkeypatch, device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs["compute_device"].type)
        raise RuntimeError(message)

    monkeypatch.setattr(resize, "resize_image_batch", fail)
    with pytest.raises(RuntimeError, match=message):
        resize.resize_minimax_image(torch.ones(1, 5, 9, 3), 13, 7, "bicubic")
    assert len(calls) == 1


def test_unchanged_dimensions_bypass_device_selection(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Identity resize must not select a device")

    monkeypatch.setattr(resize, "resolve_resize_device", forbidden)
    image = torch.zeros(1, 512, 512, 3)
    assert MAIPrepareImageForMinimaxH3().prepare(image, 0.25)[0] is image


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("algorithm", RESIZE_ALGORITHM_OPTIONS)
@pytest.mark.parametrize("input_device", ["cpu", "cuda:0"])
def test_actual_gpu_resize_matches_cpu_and_preserves_output_device(monkeypatch, algorithm, input_device):
    select_device(monkeypatch, "cpu")
    image = torch.rand(2, 7, 11, 4, device=input_device)
    expected = resize.resize_minimax_image(image, 17, 13, algorithm)
    select_device(monkeypatch, "cuda:0")
    original_resize = resize.resize_image_batch
    used_devices = []

    def record(*args, **kwargs):
        result = original_resize(*args, **kwargs)
        used_devices.append(result[0].device.type)
        return result

    monkeypatch.setattr(resize, "resize_image_batch", record)
    output = resize.resize_minimax_image(image, 17, 13, algorithm)
    assert used_devices == ["cuda"]
    assert output.device == image.device and output.dtype == image.dtype
    torch.testing.assert_close(output.cpu(), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_node_resizes_to_eight_megapixels_on_gpu(monkeypatch):
    select_device(monkeypatch, "cuda:0")
    image = torch.full((1, 16, 32, 3), 0.3)
    output, = MAIPrepareImageForMinimaxH3().prepare(image, 8.0)
    assert output.shape == (1, 2048, 4096, 3)
    assert output.device == image.device and output.dtype == image.dtype
    torch.testing.assert_close(output[:, ::128, ::128], torch.full((1, 16, 32, 3), 0.3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_input_recovery_does_not_run_gpu_memory_hooks(monkeypatch):
    select_device(monkeypatch, "cuda:0")
    image = torch.full((2, 5, 9, 3), 0.3, device="cuda:0")
    original_resize = resize.resize_image_batch
    devices = []

    def failing_gpu(source, *args, **kwargs):
        device = kwargs["compute_device"].type
        devices.append(device)
        if device == "cuda":
            raise torch.cuda.OutOfMemoryError("simulated allocation failure")
        assert source.device.type == "cpu"
        return original_resize(source, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU recovery must not run GPU memory hooks")

    monkeypatch.setattr(resize, "resize_image_batch", failing_gpu)
    monkeypatch.setattr("utils.gpu_memory.prepare_gpu_memory", forbidden)
    monkeypatch.setattr("utils.gpu_memory.release_gpu_cache", forbidden)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        output = resize.resize_minimax_image(image, 13, 7, "lanczos")
    assert devices == ["cuda", "cpu"]
    assert output.device.type == "cpu"
    torch.testing.assert_close(output, torch.full((2, 7, 13, 3), 0.3))
