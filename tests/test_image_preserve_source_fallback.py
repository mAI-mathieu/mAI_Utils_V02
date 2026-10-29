"""Node CPU execution and single-retry CUDA fallback without a ComfyUI runtime."""

import warnings

import pytest
import torch

import nodes.image_preserve_source as node_module
from nodes.image_preserve_source import MAIImagePreserveSource, _is_cuda_failure
from utils.image_preserve_source import preserve_source_image


@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_cpu_execution_never_queries_or_initializes_cuda(monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU execution touched CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    source = torch.full((1, 24, 24, 3), 0.3)
    edited = source.clone()
    edited[:, 8:16, 8:16] = 0.8
    image, mask = MAIImagePreserveSource().run(source, edited, mode=mode)
    assert image.device.type == mask.device.type == "cpu"
    assert mask[0, 12, 12] > 0.9


@pytest.mark.parametrize("message", [
    "CUDA error: no kernel image is available for execution on the device",
    "no kernel image available", "invalid device function",
    "CUDA driver version is insufficient", "CUDA initialization failed",
    "CUBLAS_STATUS_NOT_SUPPORTED", "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED",
    "Torch not compiled with CUDA enabled",
])
def test_recognizes_gpu_compatibility_failures(message):
    assert _is_cuda_failure(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "processing interrupted", "source_mask dimensions do not match",
    "Auto mode requires SciPy", "Expected all tensors to be on the same device",
    "CPU allocator failed",
])
def test_other_runtime_errors_do_not_count_as_gpu_failure(message):
    assert not _is_cuda_failure(RuntimeError(message))


def test_cpu_failure_propagates_without_retry_or_warning(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args[0].device.type)
        raise torch.OutOfMemoryError("CPU allocator failed")
    monkeypatch.setattr(node_module, "preserve_source_image", fail)
    source = torch.zeros(1, 4, 4, 3)
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(torch.OutOfMemoryError, match="CPU allocator failed"):
            MAIImagePreserveSource().run(source, source)
    assert calls == ["cpu"] and not caught


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_successful_gpu_execution_keeps_outputs_on_original_device(mode):
    source = torch.full((1, 24, 24, 3), 0.3, device="cuda")
    edited = source.clone()
    edited[:, 8:16, 8:16] = 0.8
    with warnings.catch_warnings(record=True) as caught:
        image, mask = MAIImagePreserveSource().run(source, edited, mode=mode)
    assert image.device == mask.device == source.device
    assert image.dtype == source.dtype
    assert not caught


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ["manual", "auto"])
@pytest.mark.parametrize("failure_type,message", [
    (torch.OutOfMemoryError, "CUDA out of memory"),
    (RuntimeError, "CUDA error: no kernel image is available for execution on the device"),
])
@pytest.mark.parametrize("with_mask", [False, True])
def test_gpu_failure_retries_whole_operation_on_cpu_preserving_settings_and_dtype(
    monkeypatch, mode, failure_type, message, with_mask,
):
    source = torch.linspace(0.1, 0.7, 9 * 11 * 4).reshape(1, 9, 11, 4)
    source = source.transpose(1, 2).to(device="cuda", dtype=torch.float16)
    edited = (source.cpu().double() * 0.8 + 0.1).expand(2, -1, -1, -1).clone()
    edited[:, 3:8, 3:7, :3] = 0.9
    allowed = torch.ones(source.shape[:3], device="cuda") if with_mask else None
    if allowed is not None:
        allowed[:, 0, 0] = 0
    source_before, edited_before = source.clone(), edited.clone()
    settings = dict(mode=mode, difference_threshold=0.045, transition_width=0.025,
                    comparison_blur=1.5, detail_sensitivity=0.4, mask_expand=1,
                    mask_blur=1.0, blend_mode="content_aware_seam_multiband", multiband_levels=3)
    expected = preserve_source_image(source.cpu(), edited,
        edit_mask=allowed.cpu() if allowed is not None else None, **settings)
    calls = []
    def fail_on_gpu(original_image, edited_image, edit_mask=None, **kwargs):
        calls.append(original_image.device.type)
        assert kwargs == settings
        if original_image.device.type == "cuda":
            raise failure_type(message)
        assert edited_image.device.type == "cpu"
        assert edit_mask is None or edit_mask.device.type == "cpu"
        return preserve_source_image(original_image, edited_image, edit_mask=edit_mask, **kwargs)
    monkeypatch.setattr(node_module, "preserve_source_image", fail_on_gpu)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        actual = MAIImagePreserveSource().run(source, edited, edit_mask=allowed, **settings)
    assert calls == ["cuda", "cpu"]
    assert actual[0].device.type == actual[1].device.type == "cpu"
    assert actual[0].dtype == torch.float16 and actual[1].dtype == torch.float32
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(source, source_before) and torch.equal(edited, edited_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_transfer_failure_with_cpu_source_still_gets_cpu_retry(monkeypatch):
    source = torch.zeros(1, 8, 8, 3)
    edited = torch.ones_like(source, device="cuda")
    calls = []
    def fail_first(original_image, edited_image, **kwargs):
        calls.append(edited_image.device.type)
        if edited_image.device.type == "cuda":
            raise RuntimeError("CUDA error: invalid device function")
        return preserve_source_image(original_image, edited_image, **kwargs)
    monkeypatch.setattr(node_module, "preserve_source_image", fail_first)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        image, mask = MAIImagePreserveSource().run(source, edited, mode="auto")
    assert calls == ["cuda", "cpu"]
    assert torch.equal(image, torch.ones_like(source))
    assert mask.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("failure_type,message", [
    (RuntimeError, "processing interrupted"),
    (RuntimeError, "Auto mode requires SciPy"),
    (ValueError, "dimensions must match"),
])
def test_non_hardware_gpu_errors_are_not_hidden_or_retried(monkeypatch, failure_type, message):
    source = torch.zeros(1, 4, 4, 3, device="cuda")
    calls = []
    def fail(*args, **kwargs):
        calls.append(args[0].device.type)
        raise failure_type(message)
    monkeypatch.setattr(node_module, "preserve_source_image", fail)
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(failure_type, match=message):
            MAIImagePreserveSource().run(source, source)
    assert calls == ["cuda"] and not caught


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_failed_cpu_retry_propagates_instead_of_looping(monkeypatch):
    source = torch.zeros(1, 4, 4, 3, device="cuda")
    calls = []
    def fail(*args, **kwargs):
        calls.append(args[0].device.type)
        if args[0].device.type == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory")
        raise RuntimeError("CPU processing failed")
    monkeypatch.setattr(node_module, "preserve_source_image", fail)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        with pytest.raises(RuntimeError, match="CPU processing failed"):
            MAIImagePreserveSource().run(source, source)
    assert calls == ["cuda", "cpu"]
