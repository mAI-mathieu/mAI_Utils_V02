import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from nodes import image_blur as blur_node
from nodes.image_blur import MAIImageBlur, resolve_blur_device
from utils.image_blur import blur_image


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_matches_direct_gaussian_reference_with_replicated_edges():
    image = torch.arange(18, dtype=torch.float64).reshape(1, 2, 3, 3) / 18
    original = image.clone()
    radius, sigma = 3, 1.2
    kernel = [math.exp(-0.5 * (offset / sigma) ** 2) for offset in range(-radius, radius + 1)]
    total = sum(kernel)
    kernel = [weight / total for weight in kernel]
    expected = torch.zeros_like(image)
    for y in range(2):
        for x in range(3):
            for dy, wy in zip(range(-radius, radius + 1), kernel):
                for dx, wx in zip(range(-radius, radius + 1), kernel):
                    expected[0, y, x] += image[0, min(1, max(0, y + dy)), min(2, max(0, x + dx))] * wy * wx

    result = blur_image(image, radius, sigma)

    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(image, original)


def test_impulse_is_symmetric_and_channels_and_batches_do_not_mix():
    image = torch.zeros(2, 9, 9, 3)
    image[0, 4, 4, 0] = 1
    image[1, ..., 1] = 0.5
    result = blur_image(image, radius=2, sigma=1)
    assert 0 < result[0, 4, 4, 0] < 1
    assert result[0, 4, 4, 0] > result[0, 4, 5, 0] > result[0, 4, 6, 0] > 0
    torch.testing.assert_close(result[0], result[0].flip((0, 1)))
    torch.testing.assert_close(result[0, ..., 0].sum(), torch.tensor(1.0))
    assert not result[0, ..., 1:].any()
    assert not result[1, ..., (0, 2)].any()
    torch.testing.assert_close(result[1, ..., 1], image[1, ..., 1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_preserves_dtype_shape_device_and_alpha_on_noncontiguous_inputs(dtype, channels):
    image = torch.rand(2, 5, 7, channels).to(dtype).transpose(1, 2)
    original = image.clone()
    result = blur_image(image, 2, 1)
    reference = blur_image(image.double(), 2, 1).to(dtype)
    assert result.shape == image.shape
    assert result.device == image.device
    assert result.dtype == dtype
    torch.testing.assert_close(result, reference)
    assert torch.equal(image, original)
    if channels == 4:
        assert torch.equal(result[..., 3], image[..., 3])


@pytest.mark.parametrize("height,width", [(1, 1), (1, 3), (3, 1)])
def test_large_radius_on_tiny_constant_images(height, width):
    image = torch.full((2, height, width, 3), 0.75)
    torch.testing.assert_close(blur_image(image, 128, 40), image)


def test_sigma_controls_spread_and_radius_zero_is_identity():
    image = torch.zeros(1, 7, 7, 1)
    image[0, 3, 3, 0] = 1
    assert blur_image(image, 0) is image
    narrow = blur_image(image, 3, 0.5)
    broad = blur_image(image, 3, 2)
    assert broad[0, 3, 3, 0] < narrow[0, 3, 3, 0]
    assert broad[0, 3, 5, 0] > narrow[0, 3, 5, 0]


@pytest.mark.parametrize("radius", [-1, 129, 1.5, True])
def test_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        blur_image(torch.zeros(1, 2, 3, 3), radius)


@pytest.mark.parametrize("sigma", [0, -1, math.nan, math.inf])
def test_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        blur_image(torch.zeros(1, 2, 3, 3), sigma=sigma)


@pytest.mark.parametrize("image,message", [
    (torch.zeros(2, 3, 3), "shape"),
    (torch.zeros(0, 2, 3, 3), "nonempty"),
    (torch.zeros(1, 2, 3, 2), "channels"),
    (torch.zeros(1, 2, 3, 3, dtype=torch.uint8), "float"),
])
def test_rejects_invalid_images(image, message):
    with pytest.raises(ValueError, match=message):
        blur_image(image)
    with pytest.raises(TypeError, match="tensor"):
        blur_image(None)


def test_cpu_and_auto_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    image = torch.rand(1, 4, 5, 3)
    node = MAIImageBlur()
    torch.testing.assert_close(node.run(image)[0], node.run(image, device="cpu")[0])
    with pytest.raises(ValueError, match="requires CUDA"):
        node.run(image, device="gpu")
    with pytest.raises(ValueError, match="Unknown blur device"):
        node.run(image, device="invalid")


def test_explicit_cpu_does_not_query_cuda(monkeypatch):
    def forbidden():
        raise AssertionError("CPU blur queried CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    result, = MAIImageBlur().run(torch.ones(1, 2, 3, 3), device="cpu")
    torch.testing.assert_close(result, torch.ones_like(result))


@pytest.mark.parametrize("failure", [RuntimeError("no kernel image available"), torch.OutOfMemoryError("CUDA out of memory")])
def test_auto_retries_on_cpu_after_cuda_failure(monkeypatch, failure):
    monkeypatch.setattr(blur_node, "resolve_blur_device", lambda *args: torch.device("cuda:0"))
    devices = []
    def simulate_cuda(image, radius, sigma, compute_device):
        devices.append(compute_device.type)
        if compute_device.type == "cuda":
            raise failure
        return blur_image(image, radius, sigma, compute_device)
    monkeypatch.setattr(blur_node, "blur_image", simulate_cuda)
    image = torch.rand(1, 4, 5, 3)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        result, = MAIImageBlur().run(image)
    assert devices == ["cuda", "cpu"]
    torch.testing.assert_close(result, blur_image(image))


def test_auto_retries_on_cpu_after_cuda_initialization_failure(monkeypatch):
    def fail_resolution(*args):
        raise RuntimeError("CUDA initialization failed")
    monkeypatch.setattr(blur_node, "resolve_blur_device", fail_resolution)
    image = torch.rand(1, 4, 5, 3)
    with pytest.warns(RuntimeWarning, match="initialization failed"):
        result, = MAIImageBlur().run(image)
    torch.testing.assert_close(result, blur_image(image))


@pytest.mark.parametrize("mode", ["auto", "cpu", "gpu"])
def test_cpu_errors_and_forced_gpu_errors_propagate(monkeypatch, mode):
    compute = "cpu" if mode != "gpu" else "cuda:0"
    monkeypatch.setattr(blur_node, "resolve_blur_device", lambda *args: torch.device(compute))
    def fail(*args):
        raise RuntimeError("processing failed")
    monkeypatch.setattr(blur_node, "blur_image", fail)
    with pytest.raises(RuntimeError, match="processing failed"):
        MAIImageBlur().run(torch.ones(1, 2, 3, 3), device=mode)


def test_device_selection_respects_input_and_comfy_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cuda:2")
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    assert resolve_blur_device(torch.device("cuda:1"), "auto") == torch.device("cuda:1")
    assert resolve_blur_device(torch.device("cpu"), "auto") == torch.device("cuda:2")
    assert resolve_blur_device(torch.device("cpu"), "gpu") == torch.device("cuda:2")
    management.get_torch_device = lambda: torch.device("cpu")
    assert resolve_blur_device(torch.device("cpu"), "auto").type == "cpu"
    with pytest.raises(ValueError, match="selected device"):
        resolve_blur_device(torch.device("cpu"), "gpu")


def test_import_does_not_initialize_or_query_cuda():
    code = """
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA touched at import time')
torch.cuda.is_available = forbidden
torch.cuda.current_device = forbidden
torch.cuda._lazy_init = forbidden
import nodes.image_blur
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


def test_node_defaults_and_whole_pack_registration(monkeypatch):
    assert MAIImageBlur.CATEGORY == "mAI / Image"
    assert MAIImageBlur.RETURN_TYPES == ("IMAGE",)
    assert MAIImageBlur.RETURN_NAMES == ("image",)
    inputs = MAIImageBlur.INPUT_TYPES()["required"]
    assert list(inputs) == ["image", "radius", "sigma", "device"]
    defaults = {name: spec[1]["default"] for name, spec in inputs.items() if name != "image"}
    assert defaults == {"radius": 8, "sigma": 3.0, "device": "auto"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert len(MAIImageBlur().run(torch.ones(1, 2, 3, 3), **defaults)) == 1
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "test_mai_image_blur_pack", root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIImageBlur"].__name__ == "MAIImageBlur"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIImageBlur"] == "mAI image blur"
    assert "MAIFastGPUResize" in package.NODE_CLASS_MAPPINGS
    assert "MAIMaskOutline" in package.NODE_CLASS_MAPPINGS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_cuda_matches_cpu_on_every_visible_gpu_and_preserves_device(dtype):
    image = torch.rand(2, 9, 11, 4).to(dtype)
    expected = blur_image(image, 3, 1.5)
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        # Test both common CPU ComfyUI inputs and images already on CUDA.
        actual = blur_image(image, 3, 1.5, device)
        assert actual.device == image.device and actual.dtype == dtype
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        gpu_image = image.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = blur_image(gpu_image, 3, 1.5)
        assert actual.device == device and actual.dtype == dtype
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-3)
        assert torch.equal(actual[..., 3], gpu_image[..., 3])
        # CPU fallback must return CUDA-origin images to their original GPU.
        fallback = blur_image(gpu_image, 3, 1.5, "cpu")
        assert fallback.device == device
        torch.testing.assert_close(fallback.cpu(), expected)
