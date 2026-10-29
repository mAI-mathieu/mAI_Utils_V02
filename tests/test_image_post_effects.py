"""Image finishing math and device recovery without a running ComfyUI server."""

import ast
import importlib.util
import math
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest
import torch

import nodes.image_post_effects as node_module
from nodes.image_post_effects import MAIImagePostEffects, resolve_post_effects_device
from utils.image_blur import blur_image
from utils.image_post_effects import POST_EFFECT_CONTROLS, apply_image_post_effects


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("controls", [{}, {"enabled": False, "blur": 5}, {"strength": 0, "contrast": 2}])
def test_bypass_is_exact_and_does_not_query_cuda(monkeypatch, controls):
    def forbidden(*args, **kwargs):
        raise AssertionError("Bypass touched CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    image = torch.rand(2, 4, 5, 4)
    assert apply_image_post_effects(image, **controls) is image
    assert MAIImagePostEffects().run(image, **controls)[0] is image


@pytest.mark.parametrize("controls,expected", [
    ({"contrast": 2}, [0.0, 0.0, 0.5, 1.0, 1.0]),
    ({"contrast": 0}, [0.5] * 5),
    ({"lightness": 0.2}, [0.2, 0.45, 0.7, 0.95, 1.0]),
    ({"lightness": -0.2}, [0, 0.05, 0.3, 0.55, 0.8]),
    ({"exposure": 1}, [0, 0.5, 1, 1, 1]),
    ({"gamma": 2}, [0, 0.5, math.sqrt(0.5), math.sqrt(0.75), 1]),
])
def test_tone_controls_have_known_values(controls, expected):
    image = torch.linspace(0, 1, 5, dtype=torch.float64).reshape(1, 1, 5, 1)
    result = apply_image_post_effects(image, **controls)
    torch.testing.assert_close(result.flatten(), torch.tensor(expected, dtype=torch.float64))


def test_saturation_uses_rgb_luminance_and_leaves_grayscale_alone():
    image = torch.eye(3).reshape(1, 1, 3, 3)
    result = apply_image_post_effects(image, saturation=0)
    gray = torch.tensor([0.2126, 0.7152, 0.0722]).reshape(1, 1, 3, 1).expand_as(image)
    torch.testing.assert_close(result, gray)
    mono = torch.rand(1, 3, 5, 1)
    assert torch.equal(apply_image_post_effects(mono, saturation=3), mono)


def test_order_and_strength_blend_against_independent_reference():
    image = torch.tensor([[[[0.15, 0.3, 0.65]]]], dtype=torch.float64)
    work = (image * 2 - 0.5) * 1.2 + 0.5 + 0.1
    gray = (work * torch.tensor([0.2126, 0.7152, 0.0722])).sum(-1, keepdim=True)
    expected = (gray + (work - gray) * 0.5).clamp(0, 1).sqrt()
    expected = image * 0.7 + expected * 0.3
    result = apply_image_post_effects(
        image, exposure=1, contrast=1.2, lightness=0.1, saturation=0.5, gamma=2, strength=0.3,
    )
    torch.testing.assert_close(result, expected)


def test_blur_and_sharpen_match_reference_and_do_not_mix_frames():
    image = torch.zeros(2, 9, 11, 3)
    image[0, 4, 5, 0] = 1
    image[1, ..., 1] = 0.4
    blurred = apply_image_post_effects(image, blur=1.5)
    torch.testing.assert_close(blurred, blur_image(image, 1.5))
    assert not blurred[0, ..., 1:].any()
    assert not blurred[1, ..., (0, 2)].any()
    softened = blur_image(image, 1)
    expected = (softened + 0.8 * (softened - blur_image(softened, 1))).clamp(0, 1)
    torch.testing.assert_close(apply_image_post_effects(softened, sharpen=0.8), expected)
    assert apply_image_post_effects(softened, sharpen=0.8)[0, 4, 5, 0] > softened[0, 4, 5, 0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_preserves_shape_dtype_alpha_and_input_on_noncontiguous_batches(dtype, channels):
    image = torch.rand(2, 5, 7, channels).to(dtype).transpose(1, 2)
    original = image.clone()
    controls = dict(contrast=1.2, lightness=0.05, exposure=0.25, saturation=0.7,
                    gamma=1.2, blur=0.6, sharpen=0.4, strength=0.8)
    result = apply_image_post_effects(image, **controls)
    reference = apply_image_post_effects(image.double(), **controls).to(dtype)
    torch.testing.assert_close(result, reference)
    assert result.shape == image.shape and result.dtype == dtype and result.device == image.device
    assert torch.equal(image, original)
    if channels == 4:
        assert torch.equal(result[..., 3], image[..., 3])


@pytest.mark.parametrize("height,width", [(1, 1), (1, 3), (3, 1)])
def test_large_blur_on_tiny_images_keeps_constant_colors(height, width):
    image = torch.full((2, height, width, 4), 0.4)
    result = apply_image_post_effects(image, blur=32, sharpen=3)
    torch.testing.assert_close(result, image)


@pytest.mark.parametrize("name", list(POST_EFFECT_CONTROLS))
@pytest.mark.parametrize("value", [math.nan, math.inf, True, "1", None])
def test_rejects_nonfinite_or_nonnumeric_controls(name, value):
    with pytest.raises(ValueError, match=name):
        MAIImagePostEffects().run(torch.zeros(1, 2, 3, 3), **{name: value})


@pytest.mark.parametrize("name", list(POST_EFFECT_CONTROLS))
def test_rejects_out_of_range_controls(name):
    _, minimum, maximum, _, _ = POST_EFFECT_CONTROLS[name]
    for value in (minimum - 0.01, maximum + 0.01):
        with pytest.raises(ValueError, match=name):
            apply_image_post_effects(torch.zeros(1, 2, 3, 3), **{name: value})


@pytest.mark.parametrize("image,message", [
    (torch.zeros(2, 3, 3), "shape"),
    (torch.zeros(0, 2, 3, 3), "nonempty"),
    (torch.zeros(1, 2, 3, 2), "channels"),
    (torch.zeros(1, 2, 3, 3, dtype=torch.uint8), "float"),
])
def test_rejects_invalid_images(image, message):
    with pytest.raises(ValueError, match=message):
        MAIImagePostEffects().run(image)


def test_rejects_unknown_controls_devices_and_invalid_enabled():
    image = torch.zeros(1, 2, 3, 3)
    with pytest.raises(ValueError, match="Unknown post effect controls"):
        MAIImagePostEffects().run(image, typo=1)
    with pytest.raises(ValueError, match="device"):
        MAIImagePostEffects().run(image, device="typo")
    with pytest.raises(ValueError, match="enabled"):
        MAIImagePostEffects().run(image, enabled="false")
    with pytest.raises(TypeError, match="tensor"):
        MAIImagePostEffects().run(None)


def test_explicit_cpu_and_comfy_cpu_mode_do_not_query_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU mode queried CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    image = torch.rand(1, 3, 5, 3)
    actual, = MAIImagePostEffects().run(image, blur=1, device="cpu")
    torch.testing.assert_close(actual, blur_image(image, 1))
    management = ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cpu")
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    actual, = MAIImagePostEffects().run(image, blur=1)
    torch.testing.assert_close(actual, blur_image(image, 1))


def test_device_selection_uses_selected_gpu_and_falls_back_without_cuda(monkeypatch):
    management = ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cuda:2")
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_post_effects_device(torch.device("cpu"), "auto") == torch.device("cuda:2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_post_effects_device(torch.device("cpu"), "auto").type == "cpu"
    with pytest.raises(ValueError, match="requires? CUDA"):
        resolve_post_effects_device(torch.device("cpu"), "gpu")


@pytest.mark.parametrize("failure", [
    torch.OutOfMemoryError("CUDA out of memory"), RuntimeError("no kernel image available"),
    RuntimeError("CUDA driver version is insufficient"), RuntimeError("CUDNN_STATUS_NOT_SUPPORTED"),
    NotImplementedError("Could not run 'aten::conv2d' with arguments from the 'CUDA' backend"),
])
def test_gpu_failure_retries_once_on_cpu_with_identical_settings(monkeypatch, failure):
    monkeypatch.setattr(node_module, "resolve_post_effects_device", lambda *args: torch.device("cuda:0"))
    calls = []
    def simulate(image, **kwargs):
        calls.append(kwargs.copy())
        if kwargs["compute_device"] == torch.device("cuda:0"):
            raise failure
        return apply_image_post_effects(image, **kwargs)
    monkeypatch.setattr(node_module, "apply_image_post_effects", simulate)
    image = torch.rand(2, 5, 7, 4)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        result, = MAIImagePostEffects().run(image, contrast=1.2, blur=1, strength=0.7)
    expected = apply_image_post_effects(image, contrast=1.2, blur=1, strength=0.7)
    torch.testing.assert_close(result, expected)
    assert len(calls) == 2
    assert calls[1]["compute_device"] == calls[1]["output_device"] == "cpu"
    for name in POST_EFFECT_CONTROLS:
        assert calls[0][name] == calls[1][name]


def test_cuda_initialization_failure_retries_on_cpu(monkeypatch):
    def fail(*args):
        raise RuntimeError("CUDA initialization failed")
    monkeypatch.setattr(node_module, "resolve_post_effects_device", fail)
    image = torch.rand(1, 3, 5, 3)
    with pytest.warns(RuntimeWarning, match="initialization failed"):
        actual, = MAIImagePostEffects().run(image, lightness=0.1)
    torch.testing.assert_close(actual, (image + 0.1).clamp(0, 1))


@pytest.mark.parametrize("device,message", [
    ("cpu", "CUDA out of memory"), ("auto", "processing interrupted"), ("gpu", "CUDA error: failed"),
])
def test_unrelated_cpu_and_forced_gpu_errors_propagate(monkeypatch, device, message):
    monkeypatch.setattr(node_module, "resolve_post_effects_device",
                        lambda *args: torch.device("cpu" if device == "cpu" else "cuda:0"))
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError(message)
    monkeypatch.setattr(node_module, "apply_image_post_effects", fail)
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(RuntimeError, match=message):
            MAIImagePostEffects().run(torch.zeros(1, 2, 3, 3), lightness=0.1, device=device)
    assert len(calls) == 1 and not caught


def test_node_contract_and_pack_registration_preserve_existing_nodes(monkeypatch):
    assert MAIImagePostEffects.CATEGORY == "mAI / Image"
    assert MAIImagePostEffects.RETURN_TYPES == ("IMAGE",)
    assert MAIImagePostEffects.RETURN_NAMES == ("image",)
    schema = MAIImagePostEffects.INPUT_TYPES()["required"]
    defaults = {name: spec[1]["default"] for name, spec in schema.items() if name != "image"}
    image = torch.rand(1, 2, 3, 3)
    assert MAIImagePostEffects().run(image, **defaults)[0] is image
    root = Path(__file__).resolve().parents[1]
    name = "test_mai_post_effects_pack"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIImagePostEffects"].__name__ == "MAIImagePostEffects"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIImagePostEffects"] == "mAI image post effects"
    # Every explicitly registered class remains present after the pack imports.
    tree = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
    mapping = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                           for target in node.targets))
    assert {key.value for key in mapping.keys} <= package.NODE_CLASS_MAPPINGS.keys()
    assert "MAIImageBlur" in package.NODE_CLASS_MAPPINGS
    assert "mAI_CinematicPost" in package.NODE_CLASS_MAPPINGS


def test_import_does_not_touch_cuda():
    code = """
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA touched at import')
torch.cuda.is_available = forbidden
torch.cuda.current_device = forbidden
torch.cuda._lazy_init = forbidden
import nodes.image_post_effects
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_real_cuda_matches_cpu_for_cpu_and_gpu_inputs(dtype):
    image = torch.rand(2, 9, 11, 4).to(dtype)
    settings = dict(contrast=1.2, lightness=-0.05, exposure=0.3, saturation=0.8,
                    gamma=1.1, blur=1.2, sharpen=0.5, strength=0.7)
    expected = apply_image_post_effects(image, **settings)
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        for source in (image, image.to(device)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = apply_image_post_effects(source, compute_device=device, **settings)
            assert result.device == source.device and result.dtype == dtype
            torch.testing.assert_close(result.cpu(), expected, atol=2e-3, rtol=2e-3)
            assert torch.equal(result[..., 3].cpu(), image[..., 3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_origin_fallback_returns_cpu_output_without_losing_alpha(monkeypatch):
    image = torch.rand(2, 5, 7, 4, device="cuda", dtype=torch.float16)
    real_apply = node_module.apply_image_post_effects
    monkeypatch.setattr(node_module, "resolve_post_effects_device", lambda *args: image.device)
    def fail_gpu(image, **kwargs):
        if torch.device(kwargs["compute_device"]).type == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real_apply(image, **kwargs)
    monkeypatch.setattr(node_module, "apply_image_post_effects", fail_gpu)
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        actual, = MAIImagePostEffects().run(image, lightness=0.1, blur=1)
    expected = real_apply(image.cpu(), lightness=0.1, blur=1)
    assert actual.device.type == "cpu" and actual.dtype == image.dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_node_auto_accelerates_cpu_input_on_selected_gpu_and_cpu_mode_moves_output(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    management = ModuleType("comfy.model_management")
    management.get_torch_device = lambda: device
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    real_apply = node_module.apply_image_post_effects
    compute_devices = []
    def record_compute(image, **kwargs):
        compute_devices.append(torch.device(kwargs["compute_device"]))
        return real_apply(image, **kwargs)
    monkeypatch.setattr(node_module, "apply_image_post_effects", record_compute)
    image = torch.rand(2, 32, 48, 4)
    controls = dict(contrast=1.1, blur=1.5, sharpen=0.3)
    gpu_result, = MAIImagePostEffects().run(image, **controls)
    assert compute_devices == [device]
    assert gpu_result.device.type == "cpu"
    cpu_result, = MAIImagePostEffects().run(image.to(device), device="cpu", **controls)
    assert compute_devices[-1].type == cpu_result.device.type == "cpu"
    torch.testing.assert_close(gpu_result, cpu_result, atol=2e-5, rtol=2e-5)
