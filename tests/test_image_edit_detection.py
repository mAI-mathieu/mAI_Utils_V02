"""Synthetic edit cases exercise automatic decisions without a ComfyUI server."""

import numpy as np
from pathlib import Path
import shutil
import subprocess
import pytest
import torch

from nodes.image_preserve_source import MAIImagePreserveSource
from utils.image_edit_detection import detect_image_edits, _coherent_regions
from utils.image_preserve_source import preserve_source_image


def pair(size=96, drift=0.01, change=0.35):
    original = torch.full((1, size, size, 3), 0.35)
    edited = original + drift
    start, end = size // 3, 2 * size // 3
    edited[:, start:end, start:end] += change
    return original, edited


@pytest.mark.parametrize("size", [48, 96, 192])
@pytest.mark.parametrize("drift", [0.0, 0.01, 0.04])
def test_detects_local_edit_at_multiple_resolutions_and_drift_levels(size, drift):
    original, edited = pair(size, drift)
    image, mask = preserve_source_image(original, edited, mode="auto")
    center = size // 2
    assert mask[0, center, center] == 1
    assert torch.equal(image[0, center, center], edited[0, center, center])
    assert mask[:, :size // 8].count_nonzero() == 0
    assert torch.equal(image[:, :size // 8], original[:, :size // 8])


@pytest.mark.parametrize("drift", [0.0, 0.01, 0.04])
def test_unchanged_or_uniform_small_drift_keeps_original(drift):
    original = torch.full((1, 64, 80, 3), 0.4)
    result, mask = preserve_source_image(original, original + drift, mode="auto")
    assert torch.equal(result, original)
    assert mask.count_nonzero() == 0


def test_texture_contrast_loss_does_not_select_the_whole_image():
    yy, xx = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
    texture = ((xx + yy) % 2).float() * 0.3 + 0.35
    original = texture[None, ..., None].expand(1, 96, 96, 3).clone()
    edited = torch.full_like(original, 0.5)
    edited[:, 36:60, 36:60] = 0.85
    result, mask = preserve_source_image(original, edited, mode="auto")
    assert mask[0, 48, 48] == 1
    assert mask[:, :15].count_nonzero() == 0
    assert torch.equal(result[:, :15], original[:, :15])


def test_color_only_edit_is_detected_without_geometric_change():
    original, edited = pair(change=0)
    edited[:, 32:64, 32:64, 0] = 0.8
    _, mask = preserve_source_image(original, edited, mode="auto")
    assert mask[0, 48, 48] == 1
    assert mask[:, :12].count_nonzero() == 0


def test_moved_shape_removes_old_edge_and_keeps_new_edge():
    yy, xx = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
    original = torch.full((1, 96, 96, 3), 0.2)
    edited = original.clone()
    original[0, (xx - 43).square() + (yy - 48).square() < 20 ** 2] = 0.8
    edited[0, (xx - 55).square() + (yy - 48).square() < 20 ** 2] = 0.8
    result, mask = preserve_source_image(original, edited, mode="auto")
    assert mask[0, 48, 28] == 1 and mask[0, 48, 70] == 1
    assert torch.equal(result[0, 48, 28], edited[0, 48, 28])
    assert torch.equal(result[0, 48, 70], edited[0, 48, 70])
    assert torch.equal(result[:, :12], original[:, :12])


def test_local_edit_survives_random_noise_without_selecting_background():
    generator = torch.Generator().manual_seed(71)
    original = torch.rand((1, 96, 96, 3), generator=generator) * 0.1 + 0.35
    edited = original + torch.randn(original.shape, generator=generator) * 0.025
    edited[:, 32:64, 32:64] += 0.25
    _, mask = preserve_source_image(original, edited, mode="auto")
    assert mask[0, 48, 48] == 1
    assert mask[:, :12].mean() < 0.01


def test_widespread_strong_change_retains_edit_instead_of_classifying_it_as_noise():
    original = torch.full((1, 64, 80, 3), 0.3)
    edited = torch.full_like(original, 0.7)
    result, mask = preserve_source_image(original, edited, mode="auto")
    assert torch.equal(result, edited)
    assert torch.equal(mask, torch.ones_like(mask))


def test_thresholds_adapt_to_noise_and_feather_to_resolution():
    low_source, low_edit = pair(96, drift=0)
    high_source, high_edit = pair(96, drift=0.04)
    _, low = detect_image_edits(low_source, low_edit, torch.ones(1, 96, 96))
    _, high = detect_image_edits(high_source, high_edit, torch.ones(1, 96, 96))
    large_source, large_edit = pair(192)
    _, large = detect_image_edits(large_source, large_edit, torch.ones(1, 192, 192))
    assert high.high_threshold > low.high_threshold
    assert high.low_threshold > low.low_threshold
    assert large.feather_pixels > low.feather_pixels
    assert large.multiband_levels >= low.multiband_levels


@pytest.mark.parametrize("size", [96, 192])
def test_background_drift_fades_without_tracing_the_local_edit_boundary(size):
    original, edited = pair(size, drift=0.025)
    image, mask = preserve_source_image(original, edited, mode="auto")
    # Outside the actual square edit, both inputs are flat. A near-one-pixel
    # switch of the 0.025 drift used to draw an outline around that square.
    background = image[0, size // 2, 2 * size // 3:, 0]
    assert background.diff().abs().max() < 0.008
    assert background.min() >= original[0, 0, 0, 0] - 1e-6
    assert background.max() <= edited[0, 0, 0, 0] + 1e-6
    assert mask[0, size // 2, size // 2] == 1
    assert torch.equal(image[:, :size // 8], original[:, :size // 8])


def test_hysteresis_groups_weak_changes_fills_holes_and_rejects_disconnected_noise():
    ndimage = pytest.importorskip("scipy.ndimage")
    score = np.zeros((64, 64))
    score[16:48, 16:48] = 0.04
    score[25:30, 25:30] = 0.1
    score[34:37, 34:37] = 0  # Internal gap should be filled.
    score[3:6, 3:6] = 0.04  # Weak noise without a strong seed.
    score[58, 58] = 0.2  # A single isolated strong pixel is also noise.
    regions = _coherent_regions(score, np.ones_like(score, dtype=bool), 0.03, 0.08, ndimage)
    assert regions[16:48, 16:48].all()
    assert not regions[3:6, 3:6].any()
    assert not regions[58, 58]


def test_empty_and_soft_masks_and_protected_holes_are_preserved_at_native_resolution():
    original, edited = pair()
    allowed = torch.zeros(96, 96)
    allowed[30:66, 30:66] = 0.5
    allowed[45:51, 45:51] = 0
    before = original.clone(), edited.clone(), allowed.clone()
    result, mask = preserve_source_image(original, edited, mode="auto", edit_mask=allowed)
    assert mask.max() <= 0.5
    assert mask[0, 40, 40] > 0
    assert torch.equal(result[0][allowed == 0], original[0][allowed == 0])
    for actual, expected in zip((original, edited, allowed), before):
        assert torch.equal(actual, expected)
    empty, empty_mask = preserve_source_image(original, edited, mode="auto", edit_mask=allowed * 0)
    assert torch.equal(empty, original) and empty_mask.count_nonzero() == 0


def test_batch_pairs_are_analyzed_independently_and_singletons_broadcast():
    original, edited = pair()
    batch = torch.cat((original, edited))
    result, masks = preserve_source_image(original, batch, mode="auto")
    assert torch.equal(result[0], original[0]) and masks[0].count_nonzero() == 0
    single, mask = preserve_source_image(original, edited, mode="auto")
    assert torch.equal(result[1], single[0])
    assert torch.equal(masks[1], mask[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_auto_keeps_dtype_device_and_rgba_contract(dtype):
    original, edited = pair(48)
    original = torch.cat((original, torch.full_like(original[..., :1], 0.4)), -1).to(dtype)
    edited = torch.cat((edited, torch.full_like(edited[..., :1], 0.8)), -1).to(dtype)
    result, mask = preserve_source_image(original, edited, mode="auto")
    assert result.dtype == dtype and result.device == original.device
    assert torch.equal(result[mask == 0], original[mask == 0])
    assert torch.equal(result[mask == 1], edited[mask == 1])
    assert torch.isfinite(result).all()


def test_analysis_resolution_cap_preserves_native_mask_hole():
    original = torch.zeros(1, 32, 1024, 3)
    edited = original.clone()
    edited[:, 8:24, 450:550] = 0.8
    allowed = torch.ones(32, 1024)
    allowed[:, 500] = 0
    result, mask = preserve_source_image(original, edited, mode="auto", edit_mask=allowed)
    assert mask.shape == (1, 32, 1024)
    assert mask[:, :, 500].count_nonzero() == 0
    assert torch.equal(result[:, :, 500], original[:, :, 500])
    assert mask[:, 16, 470] > 0.9


def test_auto_ignores_manual_settings_and_missing_mode_retains_legacy_result():
    original, edited = pair(48)
    automatic = preserve_source_image(original, edited, mode="auto")
    ignored = preserve_source_image(original, edited, mode="auto", difference_threshold=1,
                                     mask_expand=128, blend_mode="feather", multiband_levels=1)
    assert all(torch.equal(a, b) for a, b in zip(automatic, ignored))
    legacy = preserve_source_image(original, edited)
    manual = preserve_source_image(original, edited, mode="manual")
    assert all(torch.equal(a, b) for a, b in zip(legacy, manual))
    node = MAIImagePreserveSource()
    outputs = node.run(original, edited, mode="auto")
    assert all(torch.equal(a, b) for a, b in zip(automatic, outputs))
    assert node.INPUT_TYPES()["optional"]["mode"][1]["default"] == "manual"


def test_invalid_mode_and_missing_scipy_are_actionable(monkeypatch):
    import sys
    original, edited = pair(48)
    with pytest.raises(ValueError, match="Unknown mode"):
        preserve_source_image(original, edited, mode="invalid")
    monkeypatch.setitem(sys.modules, "scipy", None)
    with pytest.raises(RuntimeError, match="Auto mode requires SciPy"):
        preserve_source_image(original, edited, mode="auto")


def test_frontend_mode_and_saved_workflow_behavior():
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node.js unavailable for frontend tests")
    script = Path(__file__).with_name("image_preserve_source_frontend.mjs")
    result = subprocess.run([executable, str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_auto_cpu_and_cuda_agree_under_autocast():
    original, edited = pair()
    expected, expected_mask = preserve_source_image(original, edited, mode="auto")
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        with torch.autocast("cuda", dtype=torch.float16):
            actual, mask = preserve_source_image(original.to(device), edited.to(device), mode="auto")
        assert actual.device == device and mask.device == device
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(mask.cpu(), expected_mask, atol=3e-5, rtol=3e-5)
