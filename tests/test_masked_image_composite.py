import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from nodes.image_composite_masked import MAIImageCompositeMasked
from utils import masked_image_composite as composite_utils
from utils.masked_image_composite import BLEND_MODES, composite_with_source_mask


@pytest.mark.parametrize("mode", BLEND_MODES)
@pytest.mark.parametrize("channels", [3, 4])
def test_only_gray_mask_pixels_can_blend_and_inputs_are_not_modified(mode, channels):
    destination = torch.rand(2, 13, 15, channels)
    source = torch.rand_like(destination)
    mask = torch.zeros(2, 13, 15)
    mask[:, 2:11, 2:13] = 0.3
    mask[:, 4:9, 4:11] = 0.8
    mask[:, 5:8, 5:10] = 1
    mask[:, 6, 7] = 0  # Internal hole, not just the outer background.
    before = [tensor.clone() for tensor in (destination, source, mask)]

    result = composite_with_source_mask(destination, source, mask, mode)

    assert torch.equal(result[mask == 0], destination[mask == 0])
    assert torch.equal(result[mask == 1], source[mask == 1])
    for actual, original in zip((destination, source, mask), before):
        assert torch.equal(actual, original)
    assert torch.isfinite(result).all()
    if mode in ("center_cut", "content_aware_seam"):
        assert ((result == destination).all(-1) | (result == source).all(-1)).all()


def test_default_feather_direction_and_center_cut_tie_are_unambiguous():
    destination = torch.zeros(1, 1, 5, 3)
    source = torch.ones_like(destination)
    mask = torch.tensor([[0, 0.25, 0.5, 0.75, 1]])
    feather = composite_with_source_mask(destination, source, mask)
    cut = composite_with_source_mask(destination, source, mask, "center_cut")
    torch.testing.assert_close(feather[0, 0, :, 0], mask[0])
    assert torch.equal(cut[0, 0, :, 0], torch.tensor([0, 0, 1, 1, 1]))


def test_content_seam_moves_to_image_agreement_instead_of_always_using_center():
    destination = torch.zeros(1, 9, 11, 3)
    source = torch.full_like(destination, 0.9)
    source[:, :, 2:4] = 0.01  # Low disagreement corridor away from the 50% contour.
    mask = torch.linspace(0, 1, 11)[None, None].expand(1, 9, 11)
    cut = composite_with_source_mask(destination, source, mask, "center_cut")
    aware = composite_with_source_mask(destination, source, mask, "content_aware_seam")

    assert torch.equal(cut[:, :, 4], destination[:, :, 4])
    assert torch.equal(aware[:, :, 4], source[:, :, 4])
    # Only one source/destination transition per row; no detached seam islands.
    source_side = (aware == source).all(-1)
    assert torch.equal(source_side[..., 1:].int().sub(source_side[..., :-1].int()).sum(-1),
                       torch.ones(1, 9, dtype=torch.int64))
    assert (source_side[..., 1:].int() >= source_side[..., :-1].int()).all()


def test_content_seam_handles_disconnected_bands_and_holes_independently():
    destination = torch.zeros(1, 13, 19, 3)
    source = torch.ones_like(destination)
    mask = torch.zeros(1, 13, 19)
    mask[:, 1:6, 1:7] = 0.2
    mask[:, 2:5, 2:6] = 1
    mask[:, 7:12, 11:18] = 0.8
    mask[:, 8:11, 12:17] = 1
    mask[:, 9, 14] = 0
    result = composite_with_source_mask(destination, source, mask, "content_aware_seam")
    assert torch.equal(result[mask == 0], destination[mask == 0])
    assert torch.equal(result[mask == 1], source[mask == 1])
    assert ((result == 0) | (result == 1)).all()


@pytest.mark.parametrize("mode", BLEND_MODES)
@pytest.mark.parametrize("mask_value", [0, 1, 0.5])
def test_uniform_masks_and_tiny_images_are_safe(mode, mask_value):
    destination = torch.zeros(1, 1, 1, 3)
    source = torch.ones_like(destination)
    result = composite_with_source_mask(destination, source, torch.full((1, 1), mask_value), mode)
    expected = 1 if mask_value == 0.5 and mode in (
        "center_cut", "content_aware_seam", "content_aware_seam_multiband",
    ) else mask_value
    assert torch.equal(result, torch.full_like(result, expected))


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_binary_masks_bypass_seam_and_pyramid_work_without_scipy(monkeypatch, mode):
    monkeypatch.setitem(sys.modules, "scipy.sparse", None)
    destination = torch.rand(1, 4, 5, 3)
    source = torch.rand_like(destination)
    mask = torch.randint(2, (1, 4, 5))
    result = composite_with_source_mask(destination, source, mask, mode)
    assert torch.equal(result, torch.where(mask[..., None] == 1, source, destination))


def test_multiband_one_level_is_feather_and_more_levels_change_the_feather_only():
    torch.manual_seed(21)
    destination = torch.rand(1, 17, 19, 3) * 0.4
    source = torch.rand_like(destination) * 0.4 + 0.5
    mask = torch.linspace(0, 1, 19)[None, None].expand(1, 17, 19)
    feather = composite_with_source_mask(destination, source, mask)
    one = composite_with_source_mask(destination, source, mask, "multiband", 1)
    five = composite_with_source_mask(destination, source, mask, "multiband", 5)
    assert torch.equal(one, feather)
    assert not torch.allclose(five, feather)
    assert torch.equal(five[mask == 0], destination[mask == 0])
    assert torch.equal(five[mask == 1], source[mask == 1])


def test_multiband_rgba_alpha_uses_supplied_feather_without_pyramid_corrections():
    destination = torch.rand(1, 7, 9, 4)
    source = torch.rand_like(destination)
    mask = torch.rand(7, 9)
    result = composite_with_source_mask(destination, source, mask, "multiband")
    expected = torch.lerp(destination[..., 3], source[..., 3], mask)
    assert torch.equal(result[..., 3], expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_combined_one_level_is_exactly_the_content_aware_cut(dtype):
    destination = torch.rand(2, 9, 11, 4, dtype=dtype)
    source = torch.rand_like(destination)
    mask = torch.linspace(0, 1, 11)[None, None].expand(1, 9, 11)
    cut = composite_with_source_mask(destination, source, mask, "content_aware_seam")
    combined = composite_with_source_mask(destination, source, mask, "content_aware_seam_multiband", 1)
    assert torch.equal(combined, cut)


def test_combined_mode_follows_the_optimized_seam_and_smooths_its_transition():
    destination = torch.zeros(1, 9, 11, 3)
    source = torch.full_like(destination, 0.9)
    source[:, :, 2:4] = 0.01
    mask = torch.linspace(0, 1, 11)[None, None].expand(1, 9, 11)
    cut = composite_with_source_mask(destination, source, mask, "content_aware_seam")
    multiband = composite_with_source_mask(destination, source, mask, "multiband", 3)
    combined = composite_with_source_mask(destination, source, mask, "content_aware_seam_multiband", 3)

    assert not torch.allclose(combined, cut)
    assert not torch.allclose(combined, multiband)
    # The seam lies left of mask 0.5. Combined blending must keep that shift,
    # while smoothing the hard source selection at the right of the corridor.
    assert (combined[:, :, 4] > multiband[:, :, 4]).all()
    assert ((combined[:, :, 4] > destination[:, :, 4]) & (combined[:, :, 4] < source[:, :, 4])).all()
    assert torch.equal(combined[mask == 0], destination[mask == 0])
    assert torch.equal(combined[mask == 1], source[mask == 1])


def test_combined_rgba_alpha_follows_the_hard_seam_while_rgb_is_smoothed():
    destination = torch.zeros(1, 9, 11, 4)
    source = torch.ones_like(destination)
    mask = torch.linspace(0, 1, 11)[None, None].expand(1, 9, 11)
    cut = composite_with_source_mask(destination, source, mask, "content_aware_seam")
    combined = composite_with_source_mask(destination, source, mask, "content_aware_seam_multiband")
    assert torch.equal(combined[..., 3], cut[..., 3])
    assert not torch.allclose(combined[..., :3], cut[..., :3])


@pytest.mark.parametrize("mode", BLEND_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtypes_noncontiguous_inputs_and_identical_image_roundtrip(mode, dtype):
    image = torch.rand(1, 9, 7, 4, dtype=dtype).transpose(1, 2)
    mask = torch.rand(7, 9)
    result = composite_with_source_mask(image, image, mask, mode)
    assert result.dtype == dtype
    assert result.device == image.device
    assert torch.equal(result, image)


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_singleton_batches_broadcast_across_images_or_masks_and_mask_is_clamped(mode):
    destination = torch.zeros(1, 3, 4, 3)
    source = torch.stack([torch.ones(3, 4, 3), torch.full((3, 4, 3), 0.7)])
    mask = torch.tensor([[-1, 0.2, 0.8, 2]]).expand(3, 4)
    result = composite_with_source_mask(destination, source, mask, mode)
    assert result.shape == (2, 3, 4, 3)
    for index in range(2):
        single = composite_with_source_mask(destination, source[index:index + 1], mask, mode)
        assert torch.equal(result[index:index + 1], single)
    assert torch.equal(result[:, :, 0], destination.expand_as(result)[:, :, 0])
    assert torch.equal(result[:, :, -1], source[:, :, -1])
    masks = torch.stack([torch.zeros(3, 4), torch.ones(3, 4)])
    result = composite_with_source_mask(destination, source[:1], masks, mode)
    assert torch.equal(result[0], destination[0])
    assert torch.equal(result[1], source[0])


@pytest.mark.parametrize("mode", ["content_aware_seam", "content_aware_seam_multiband"])
def test_reduced_graph_search_cannot_change_native_hard_endpoints(monkeypatch, mode):
    monkeypatch.setattr(composite_utils, "MAX_SEAM_PIXELS", 20)
    original_cut = composite_utils._graph_cut
    sizes = []

    def checked_cut(alpha, difference):
        sizes.append(int(((alpha > 0) & (alpha < 1)).sum()))
        return original_cut(alpha, difference)

    monkeypatch.setattr(composite_utils, "_graph_cut", checked_cut)
    destination = torch.rand(1, 15, 19, 3)
    source = torch.rand_like(destination)
    mask = torch.rand(1, 15, 19)
    mask[:, 0, :] = 0
    mask[:, -1, :] = 1
    mask[:, 7, 9] = 0  # A native one-pixel hole must survive coarse analysis.
    result = composite_with_source_mask(destination, source, mask, mode)
    assert sizes and sizes[0] <= 20
    assert torch.equal(result[mask == 0], destination[mask == 0])
    assert torch.equal(result[mask == 1], source[mask == 1])


@pytest.mark.parametrize("mode", ["content_aware_seam", "content_aware_seam_multiband"])
def test_missing_scipy_reports_an_actionable_error_only_for_content_seam(monkeypatch, mode):
    monkeypatch.setitem(sys.modules, "scipy.sparse", None)
    image = torch.rand(1, 3, 5, 3)
    with pytest.raises(RuntimeError, match="Install scipy"):
        composite_with_source_mask(image, 1 - image, torch.full((3, 5), 0.5), mode)
    for mode in ("feather", "center_cut", "multiband"):
        composite_with_source_mask(image, 1 - image, torch.full((3, 5), 0.5), mode)


@pytest.mark.parametrize("change,message", [
    ("image_type", "floating RGB/RGBA"), ("size", "dimensions and channels"),
    ("channels", "dimensions and channels"), ("mask_size", "spatial dimensions"),
    ("batch", "batch sizes"), ("nan_image", "finite"), ("nan_mask", "finite"),
    ("mode", "blend_mode"), ("levels_zero", "multiband_levels"),
    ("levels_float", "multiband_levels"), ("levels_bool", "multiband_levels"),
])
def test_invalid_inputs_raise_clear_errors(change, message):
    kwargs = dict(destination_image=torch.zeros(2, 4, 5, 3),
                  source_image=torch.ones(1, 4, 5, 3), source_mask=torch.ones(4, 5))
    if change == "image_type":
        kwargs["source_image"] = torch.ones(1, 4, 5, 3, dtype=torch.int32)
    elif change == "size":
        kwargs["source_image"] = torch.ones(1, 4, 6, 3)
    elif change == "channels":
        kwargs["source_image"] = torch.ones(1, 4, 5, 4)
    elif change == "mask_size":
        kwargs["source_mask"] = torch.ones(5, 4)
    elif change == "batch":
        kwargs["source_mask"] = torch.ones(3, 4, 5)
    elif change == "nan_image":
        kwargs["source_image"].fill_(float("nan"))
    elif change == "nan_mask":
        kwargs["source_mask"].fill_(float("nan"))
    elif change == "mode":
        kwargs["blend_mode"] = "unknown"
    else:
        kwargs["multiband_levels"] = {"levels_zero": 0, "levels_float": 1.5, "levels_bool": True}[change]
    with pytest.raises(ValueError, match=message):
        composite_with_source_mask(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", BLEND_MODES)
def test_cuda_preserves_device_and_endpoints_with_cpu_mask_and_autocast(mode):
    destination = torch.rand(1, 13, 15, 3, device="cuda", dtype=torch.float16)
    source = torch.rand_like(destination)
    mask = torch.rand(1, 13, 15)
    mask[:, 0] = 0
    mask[:, -1] = 1
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = composite_with_source_mask(destination, source, mask, mode)
    assert result.device == destination.device and result.dtype == destination.dtype
    assert torch.equal(result[:, 0], destination[:, 0])
    assert torch.equal(result[:, -1], source[:, -1])
    assert torch.isfinite(result).all()


def test_node_contract_default_optional_setting_and_existing_pack_registrations(monkeypatch):
    node = MAIImageCompositeMasked()
    required = node.INPUT_TYPES()["required"]
    assert list(required) == ["destination_image", "source_image", "source_mask", "blend_mode"]
    assert required["source_mask"][0] == "MASK"
    assert required["blend_mode"][0] == list(BLEND_MODES)
    assert list(BLEND_MODES) == ["feather", "content_aware_seam", "center_cut", "multiband",
                               "content_aware_seam_multiband"]
    assert node.CATEGORY == "mAI / Image"
    assert node.RETURN_TYPES == ("IMAGE",) and node.RETURN_NAMES == ("image",)
    assert torch.equal(node.run(torch.zeros(1, 2, 3, 3), torch.ones(1, 2, 3, 3),
                                torch.ones(2, 3))[0], torch.ones(1, 2, 3, 3))

    repo = Path(__file__).resolve().parents[1]
    package_name = "test_mai_masked_composite_pack"
    spec = importlib.util.spec_from_file_location(package_name, repo / "__init__.py",
                                                 submodule_search_locations=[str(repo)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIImageCompositeMasked"].__name__ == "MAIImageCompositeMasked"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIImageCompositeMasked"] == "mAI image composite masked"
    for key in ("MAICompositeLayer", "MAIMaskSmartStitch", "MAIMaskFill"):
        assert key in package.NODE_CLASS_MAPPINGS
