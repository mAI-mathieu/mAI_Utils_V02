import sys
import subprocess
from types import ModuleType

import pytest
import torch

from nodes.fast_gpu_resize import MAIFastGPUResize, resolve_compute_device


def test_import_does_not_query_or_initialize_cuda():
    code = """
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA touched at import time')
torch.cuda.is_available = forbidden
torch.cuda.current_device = forbidden
torch.cuda._lazy_init = forbidden
import nodes.fast_gpu_resize
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


def test_node_contract_and_defaults():
    node = MAIFastGPUResize()
    inputs = node.INPUT_TYPES()["required"]
    assert node.CATEGORY == "mAI / Image"
    assert node.RETURN_TYPES == ("IMAGE", "INT", "INT", "STRING")
    assert node.RETURN_NAMES == ("image", "width", "height", "method_used")
    assert inputs["multiple_of"][0] == "INT"
    assert inputs["multiple_of"][1]["min"] == 1
    assert inputs["resize_mode"][0] == ["exact", "keep_aspect_fit", "keep_aspect_fill", "keep_aspect"]
    assert inputs["resize_mode"][1]["default"] == "exact"
    defaults = {name:spec[1]["default"] for name,spec in inputs.items() if name != "image"}
    image = torch.ones(1,720,1280,3)
    result,w,h,used = node.run(image, **defaults)
    assert result is image and (w,h,used) == (1280,720,"identity")
    # A number outside the former presets also works as a connected INT value.
    result,w,h,_ = node.run(torch.ones(1,4,8,3),25,26,method="nearest",multiple_of=3)
    assert result.shape == (1,27,24,3) and (w,h) == (24,27)


def test_node_keep_aspect_returns_actual_dimensions_without_padding():
    result, w, h, used = MAIFastGPUResize().run(
        torch.ones(2, 9, 16, 3), 32, 32, resize_mode="keep_aspect", method="nearest",
    )
    assert result.shape == (2, 18, 32, 3)
    assert (w, h, used) == (32, 18, "nearest")
    assert result.eq(1).all()


@pytest.mark.parametrize("precision,dtype", [("fp32",torch.float32),("fp16",torch.float16),("bf16",torch.bfloat16)])
def test_explicit_precision_and_cpu_are_respected_on_identity(precision,dtype):
    result,w,h,used = MAIFastGPUResize().run(torch.ones(1,3,4,3),4,3,device="cpu",precision=precision)
    assert result.dtype == dtype and result.device.type == "cpu"
    assert (w,h,used) == (4,3,"identity")


def test_auto_keeps_current_device_and_gpu_error_is_clear(monkeypatch):
    assert resolve_compute_device(torch.device("cpu"),"auto") == torch.device("cpu")
    assert resolve_compute_device(torch.device("cuda:1"),"auto") == torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(ValueError,match="available CUDA"):
        resolve_compute_device(torch.device("cpu"),"gpu")
    with pytest.raises(ValueError,match="Unknown device"):
        resolve_compute_device(torch.device("cpu"),"fake")


def test_explicit_gpu_uses_comfy_device_helper(monkeypatch):
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    management.get_torch_device = lambda:torch.device("cuda:2")
    monkeypatch.setitem(sys.modules,"comfy",comfy)
    monkeypatch.setitem(sys.modules,"comfy.model_management",management)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    assert resolve_compute_device(torch.device("cpu"),"gpu") == torch.device("cuda:2")
    management.get_torch_device = lambda:torch.device("cpu")
    with pytest.raises(ValueError,match="preferred device is not CUDA"):
        resolve_compute_device(torch.device("cpu"),"gpu")


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_explicit_device_transfers_and_auto_keeps_cuda():
    image = torch.rand(2,8,12,3)
    gpu = MAIFastGPUResize().run(image,6,4,device="gpu")[0]
    assert gpu.device.type == "cuda"
    identity = MAIFastGPUResize().run(gpu,6,4)[0]
    assert identity is gpu
    cpu = MAIFastGPUResize().run(gpu,6,4,device="cpu")[0]
    assert cpu.device.type == "cpu"
    torch.testing.assert_close(cpu,gpu.cpu())


def test_identity_and_cpu_skip_gpu_memory_management(monkeypatch):
    import utils.gpu_memory as memory
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA memory management touched")
    monkeypatch.setattr(memory, "prepare_gpu_memory", forbidden)
    monkeypatch.setattr(memory, "release_gpu_cache", forbidden)
    image = torch.ones(2,8,12,3)
    assert MAIFastGPUResize().run(image,12,8)[0] is image
    assert MAIFastGPUResize().run(image,6,4)[0].shape == (2,4,6,3)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_resize_returns_unused_vram_to_driver_and_preserves_live_tensors():
    image = torch.rand(2,32,48,3,device="cuda")
    before = image.cpu()
    scratch = torch.empty(32*1024*1024,device="cuda")
    del scratch
    result = MAIFastGPUResize().run(image,24,16,method="lanczos3")[0]
    assert result.device == image.device and result.shape == (2,16,24,3)
    torch.testing.assert_close(image.cpu(),before)
    assert torch.cuda.memory_reserved() - torch.cuda.memory_allocated() < 32*1024*1024
