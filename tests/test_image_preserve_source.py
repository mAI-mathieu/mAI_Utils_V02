import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from nodes.image_preserve_source import MAIImagePreserveSource
from utils.image_preserve_source import preserve_source_image
from utils.masked_image_composite import BLEND_MODES, composite_with_source_mask


def raw_selection(original, edited, **kwargs):
    settings = dict(comparison_blur=0, transition_width=0, mask_expand=0, mask_blur=0)
    settings.update(kwargs)
    return preserve_source_image(original, edited, **settings)


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_low_drift_restored_large_edit_kept_exactly(mode):
    original = torch.full((1, 5, 7, 3), 0.4)
    edited = original + 0.02
    edited[:, 1:4, 2:5] = 0.9
    before = original.clone(), edited.clone()
    result, mask = raw_selection(original, edited, blend_mode=mode)
    assert torch.equal(result[mask == 0], original[mask == 0])
    assert torch.equal(result[mask == 1], edited[mask == 1])
    assert mask.sum() == 9
    assert torch.equal(original, before[0]) and torch.equal(edited, before[1])


def test_default_multiscale_detection_rejects_texture_drift_but_keeps_broad_edit():
    original = torch.full((1, 49, 49, 3), 0.5)
    yy, xx = torch.meshgrid(torch.arange(49), torch.arange(49), indexing="ij")
    drift = (((xx + yy) % 2) * 2 - 1) * 0.12
    edited = original + drift[None, ..., None]
    edited[:, 17:32, 17:32] = 0.9
    result, mask = preserve_source_image(original, edited)
    assert mask[0, 24, 24] > 0.99
    assert torch.equal(result[:, :5], original[:, :5])
    assert mask[:, :5].count_nonzero() == 0
    _, raw_mask = preserve_source_image(original, edited, detail_sensitivity=1)
    assert raw_mask[:, :5].mean() > 0.95


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_identical_images_and_full_change_have_safe_defaults(mode):
    original = torch.full((1, 1, 1, 3), 0.25)
    unchanged, empty = preserve_source_image(original, original, blend_mode=mode)
    assert torch.equal(unchanged, original) and empty.count_nonzero() == 0
    changed, full = preserve_source_image(original, torch.ones_like(original), blend_mode=mode)
    torch.testing.assert_close(changed, torch.ones_like(original))
    torch.testing.assert_close(full, torch.ones_like(full))


def test_threshold_boundary_and_smoothstep_direction():
    original = torch.zeros(1, 1, 5, 3, dtype=torch.float64)
    values = torch.tensor([0.125, 0.25, 0.375, 0.5, 0.625], dtype=torch.float64)
    edited = values[None, None, :, None].expand_as(original)
    image, mask = raw_selection(original, edited, difference_threshold=0.25,
                                transition_width=0.25, blend_mode="feather")
    torch.testing.assert_close(mask[0, 0], torch.tensor([0, 0, 0.5, 1, 1], dtype=mask.dtype))
    torch.testing.assert_close(image, edited * mask[..., None])
    _, hard = raw_selection(original, edited, difference_threshold=0.25)
    assert torch.equal(hard[0, 0], torch.tensor([0, 0, 1, 1, 1], dtype=mask.dtype))
    _, none = raw_selection(original, edited, difference_threshold=1)
    assert none.count_nonzero() == 0


def test_expansion_and_feathering_have_bounded_support():
    original = torch.zeros(1, 31, 31, 3)
    edited = original.clone()
    edited[:, 15, 15] = 1
    _, expanded = raw_selection(original, edited, mask_expand=2)
    assert expanded.sum() == 25
    _, feather = raw_selection(original, edited, mask_expand=2, mask_blur=1)
    assert feather[0, 15, 18] > 0
    assert feather[:, :10].count_nonzero() == 0
    assert ((feather > 0) & (feather < 1)).any()


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_optional_mask_protects_black_even_after_expansion_and_seam(mode):
    original = torch.full((1, 17, 19, 3), 0.25)
    edited = torch.full_like(original, 0.9)
    allowed = torch.zeros(17, 19)
    allowed[3:14, 4:15] = 1
    allowed[7:10, 8:11] = 0  # Protected hole inside the edit.
    result, mask = preserve_source_image(original, edited, edit_mask=allowed, blend_mode=mode)
    assert torch.equal(result[0][allowed == 0], original[0][allowed == 0])
    assert mask[0][allowed == 0].count_nonzero() == 0


def test_soft_optional_mask_attenuates_once_and_singletons_broadcast():
    original = torch.zeros(1, 2, 3, 3)
    edited = torch.ones(1, 2, 3, 3)
    allowed = torch.stack((torch.full((2, 3), 0.5), torch.ones(2, 3)))
    result, mask = raw_selection(original, edited, edit_mask=allowed, blend_mode="feather")
    assert result.shape == (2, 2, 3, 3)
    assert torch.equal(mask, allowed)
    assert torch.equal(result[..., 0], allowed)
    _, clamped = raw_selection(original, edited, edit_mask=torch.tensor([[-1, 2, 1], [-1, 2, 1]]))
    assert torch.equal(clamped[0], torch.tensor([[0., 1, 1], [0, 1, 1]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_noncontiguous_rgba_and_batch_contract(dtype):
    original = torch.rand(1, 9, 11, 4).to(dtype).transpose(1, 2)
    edited = torch.rand(2, 9, 11, 4).to(dtype).transpose(1, 2)
    result, mask = preserve_source_image(original, edited)
    assert result.shape == edited.shape
    assert result.dtype == dtype and result.device == original.device
    assert mask.shape == edited.shape[:3] and mask.device == original.device
    assert torch.isfinite(result).all() and ((mask >= 0) & (mask <= 1)).all()
    # Alpha is composited using the same mask, but never drives detection.
    alpha_only = original.clone()
    alpha_only[..., 3] = 1 - original[..., 3]
    same, empty = preserve_source_image(original, alpha_only)
    assert torch.equal(same, original) and empty.count_nonzero() == 0


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_modes_use_shared_compositor_and_protect_native_endpoints(mode):
    torch.manual_seed(7)
    original = torch.rand(1, 17, 19, 3) * 0.3
    edited = torch.rand_like(original) * 0.3 + 0.5
    allowed = torch.linspace(0, 1, 19)[None, None].expand(1, 17, 19)
    actual, mask = raw_selection(original, edited, blend_mode=mode, edit_mask=allowed)
    expected = composite_with_source_mask(original, edited, mask, mode, 5)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[mask == 0], original[mask == 0])
    assert torch.equal(actual[mask == 1], edited[mask == 1])


@pytest.mark.parametrize("options, message", [
    ({"difference_threshold": -0.1}, "difference_threshold"),
    ({"difference_threshold": float("nan")}, "difference_threshold"),
    ({"transition_width": 2}, "transition_width"),
    ({"comparison_blur": float("inf")}, "comparison_blur"),
    ({"detail_sensitivity": -1}, "detail_sensitivity"),
    ({"mask_expand": 1.5}, "mask_expand"),
    ({"mask_expand": True}, "mask_expand"),
    ({"mask_blur": 33}, "mask_blur"),
    ({"blend_mode": "unknown"}, "blend_mode"),
    ({"multiband_levels": 0}, "multiband_levels"),
])
def test_invalid_settings_raise_clear_errors(options, message):
    image = torch.zeros(1, 2, 3, 3)
    with pytest.raises(ValueError, match=message):
        preserve_source_image(image, image, **options)


@pytest.mark.parametrize("original, edited, mask, message", [
    (torch.zeros(1, 2, 3, 3), torch.zeros(1, 3, 3, 3), None, "dimensions"),
    (torch.zeros(2, 2, 3, 3), torch.zeros(3, 2, 3, 3), None, "batch"),
    (torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 3, 3), torch.ones(4, 3), "edit_mask"),
    (torch.zeros(2, 2, 3, 3), torch.zeros(2, 2, 3, 3), torch.ones(3, 2, 3), "batch"),
    (torch.zeros(1, 0, 3, 3), torch.zeros(1, 0, 3, 3), None, "non-empty"),
    (torch.zeros(1, 2, 3, 3, dtype=torch.int32), torch.zeros(1, 2, 3, 3), None, "floating"),
    (torch.full((1, 2, 3, 3), float("nan")), torch.zeros(1, 2, 3, 3), None, "finite"),
    (torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 3, 3), torch.full((2, 3), float("nan")), "finite"),
])
def test_invalid_inputs_raise_clear_errors(original, edited, mask, message):
    with pytest.raises(ValueError, match=message):
        preserve_source_image(original, edited, edit_mask=mask)


def test_node_defaults_and_pack_registration(monkeypatch):
    node = MAIImagePreserveSource
    assert node.RETURN_TYPES == ("IMAGE", "MASK")
    assert node.RETURN_NAMES == ("image", "change_mask")
    assert node.CATEGORY == "mAI / Image"
    inputs = node.INPUT_TYPES()["required"]
    settings = {name: spec[1]["default"] for name, spec in inputs.items() if len(spec) > 1}
    original = torch.zeros(1, 3, 3, 3)
    image, mask = node().run(original, original, **settings)
    assert torch.equal(image, original) and mask.count_nonzero() == 0
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "test_mai_preserve_source_pack", root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIImagePreserveSource"].__name__ == node.__name__
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIImagePreserveSource"] == "mAI image preserve source"
    assert "MAIImageCompositeMasked" in package.NODE_CLASS_MAPPINGS
    assert "MAIBackgroundLightingMatch" in package.NODE_CLASS_MAPPINGS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_under_autocast():
    torch.manual_seed(12)
    original = torch.rand(1, 17, 19, 3)
    edited = torch.rand_like(original)
    expected, expected_mask = preserve_source_image(original, edited)
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        with torch.autocast("cuda", dtype=torch.float16):
            actual, mask = preserve_source_image(original.to(device), edited.to(device))
        assert actual.device == device and mask.device == device
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(mask.cpu(), expected_mask, atol=2e-5, rtol=2e-5)
