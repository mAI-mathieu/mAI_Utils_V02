"""Check detail preservation, conservative recovery and large-edit decisions."""

import math

import pytest
import torch

from utils.image_blur import blur_image
from utils.image_edit_blend import blend_edit_preserving_details, recover_matching_detail
from utils.image_edit_detection import detect_image_edits
from utils.image_preserve_source import preserve_source_image


def textured_image(size=96):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    pattern = 0.5 + 0.08 * torch.sin(xx * math.pi / 3) + 0.06 * torch.sin(yy * math.pi / 5)
    return pattern[None, ..., None].expand(1, size, size, 3).clone()


def detail_energy(image):
    detail = image - blur_image(image, 1)
    return detail[:, 12:-12, 12:-12].square().mean().sqrt()


def test_transition_keeps_fine_contrast_while_blending_broad_color():
    source = textured_image()
    edited = source.roll(2, dims=2) + 0.15
    mask = torch.full(source.shape[:3], 0.5)
    result = blend_edit_preserving_details(source, edited, mask, levels=4)
    assert detail_energy(result) >= 0.95 * detail_energy(edited)
    assert abs(result[:, 12:-12, 12:-12].mean() - (source + edited)[:, 12:-12, 12:-12].mean() / 2) < 0.01


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_sharp_blend_keeps_exact_native_endpoints_alpha_and_inputs(dtype):
    source = torch.rand(1, 25, 31, 4).to(dtype).transpose(1, 2)
    edited = torch.rand_like(source)
    mask = torch.linspace(0, 1, source.shape[2])[None, None].expand(*source.shape[:3])
    before = source.clone(), edited.clone(), mask.clone()
    result = blend_edit_preserving_details(source, edited, mask, 4)
    assert result.dtype == dtype and result.device == source.device
    assert torch.equal(result[mask == 0], source[mask == 0])
    assert torch.equal(result[mask == 1], edited[mask == 1])
    assert ((result[..., 3] == source[..., 3]) | (result[..., 3] == edited[..., 3])).all()
    assert torch.isfinite(result).all()
    for tensor, saved in zip((source, edited, mask), before):
        assert torch.equal(tensor, saved)


def test_detail_recovery_improves_aligned_blurred_texture_without_restoring_daylight():
    source = textured_image()
    edited = blur_image(source, 0.9) * 0.35
    result = recover_matching_detail(source, edited, torch.ones(source.shape[:3]))
    assert detail_energy(result) > 1.1 * detail_energy(edited)
    assert abs(result.mean() - edited.mean()) < 0.001
    assert result.max() < 0.3
    assert (result - edited).abs().max() <= 0.020001


def test_detail_recovery_rejects_displaced_texture_flat_replacements_and_soft_masks():
    source = textured_image()
    moved = blur_image(source.roll(3, dims=2), 0.9) * 0.35
    restored = recover_matching_detail(source, moved, torch.ones(source.shape[:3]))
    assert torch.equal(restored[:, 12:-12, 12:-12], moved[:, 12:-12, 12:-12])
    flat = torch.full_like(source, 0.25)
    assert torch.equal(recover_matching_detail(source, flat, torch.ones(source.shape[:3])), flat)
    blurred = blur_image(source, 0.9)
    assert torch.equal(recover_matching_detail(source, blurred, torch.full(source.shape[:3], 0.5)), blurred)


def test_detail_recovery_uses_edited_color_and_keeps_alpha():
    source = textured_image()
    edited = blur_image(source, 0.9) * source.new_tensor([0.8, 0.3, 0.2])
    source = torch.cat((source, torch.ones_like(source[..., :1])), -1)
    edited = torch.cat((edited, torch.full_like(edited[..., :1], 0.6)), -1)
    result = recover_matching_detail(source, edited, torch.ones(source.shape[:3]))
    assert torch.equal(result[..., 3], edited[..., 3])
    torch.testing.assert_close(result[..., 0] / result[..., 1], edited[..., 0] / edited[..., 1])
    assert detail_energy(result[..., :3]) > detail_energy(edited[..., :3])


@pytest.mark.parametrize("seed", [17, 29])
def test_unrelated_images_keep_edit_exactly_instead_of_rebuilding_source(seed):
    generator = torch.Generator().manual_seed(seed)
    source = torch.rand((1, 96, 96, 3), generator=generator)
    edited = torch.rand((1, 96, 96, 3), generator=generator)
    mask, settings = detect_image_edits(source, edited, torch.ones(source.shape[:3]))
    assert settings.strategy == "edited"
    assert torch.equal(mask, torch.ones_like(mask))
    result, _ = preserve_source_image(source, edited, mode="auto")
    assert torch.equal(result, edited)


def test_large_background_replacement_is_not_misclassified_as_noise():
    generator = torch.Generator().manual_seed(33)
    source = torch.rand((1, 96, 96, 3), generator=generator)
    edited = torch.rand((1, 96, 96, 3), generator=generator)
    edited[:, 50:85, 20:75] = source[:, 50:85, 20:75]
    result, mask = preserve_source_image(source, edited, mode="auto")
    assert torch.equal(result, edited)
    assert torch.equal(mask, torch.ones_like(mask))


def test_flat_replacement_does_not_claim_correspondence_with_textured_source():
    source = textured_image()
    edited = torch.full_like(source, 0.5)
    _, settings = detect_image_edits(source, edited, torch.ones(source.shape[:3]))
    assert settings.strategy == "edited"
    result, _ = preserve_source_image(source, edited, mode="auto")
    assert torch.equal(result, edited)


def test_real_edit_touching_image_border_keeps_its_solid_core():
    source = torch.full((1, 96, 96, 3), 0.3)
    edited = source.clone()
    edited[:, :25, :25] = 0.8
    result, mask = preserve_source_image(source, edited, mode="auto")
    assert mask[0, 0, 0] == 1 and mask[0, 12, 12] == 1
    assert torch.equal(result[0, 0, 0], edited[0, 0, 0])
    assert torch.equal(result[:, 50:], source[:, 50:])


def test_large_matching_relight_recovers_detail_and_respects_protection_mask():
    source = textured_image()
    edited = blur_image(source, 0.9) * 0.35
    _, settings = detect_image_edits(source, edited, torch.ones(source.shape[:3]))
    assert settings.strategy == "global_detail"
    allowed = torch.ones(source.shape[:3])
    allowed[:, 35:45, 35:45] = 0
    result, mask = preserve_source_image(source, edited, mode="auto", edit_mask=allowed)
    assert torch.equal(result[allowed == 0], source[allowed == 0])
    assert torch.equal(mask, allowed)
    expected = recover_matching_detail(source, edited, allowed)
    assert torch.equal(result[allowed == 1], expected[allowed == 1])


def test_mixed_batch_uses_each_pairs_strategy_and_handles_converted_dtype():
    source = textured_image()
    generator = torch.Generator().manual_seed(35)
    edits = torch.cat((source, blur_image(source, 0.9) * 0.35,
                       torch.rand(source.shape, generator=generator)))
    result, masks = preserve_source_image(source, edits.double(), mode="auto")
    for index in range(3):
        single, mask = preserve_source_image(source, edits[index:index + 1].double(), mode="auto")
        assert torch.equal(result[index], single[0])
        assert torch.equal(masks[index], mask[0])
    assert result.dtype == source.dtype
    assert torch.equal(result[0], source[0])
    assert torch.equal(result[2], edits[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_detail_operations_match_cpu_on_cuda_under_autocast():
    source = textured_image()
    edited = blur_image(source, 0.9) * 0.35
    mask = torch.ones(source.shape[:3])
    expected = recover_matching_detail(source, edited, mask)
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        with torch.autocast("cuda", dtype=torch.float16):
            actual = recover_matching_detail(source.to(device), edited.to(device), mask.to(device))
        assert actual.device == device and actual.dtype == source.dtype
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
