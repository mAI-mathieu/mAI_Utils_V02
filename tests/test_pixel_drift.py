"""Synthetic registration cases; no server, models or image downloads needed."""

import importlib.util
from pathlib import Path
import sys
import shutil
import subprocess

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")

from nodes.pixel_drift_fix import MAIPixelDriftFix
from utils.pixel_drift import _safe_transform, fix_pixel_drift


def scene(seed=4, size=256):
    rng = np.random.default_rng(seed)
    pixels = cv2.GaussianBlur(rng.random((size, size, 3)).astype(np.float32), (0, 0), 1)
    pixels = np.clip((pixels - 0.5) * 2 + 0.5, 0, 1)
    for _ in range(65):
        x, y = rng.integers(8, size - 8, 2)
        color = tuple(float(v) for v in rng.random(3))
        cv2.circle(pixels, (int(x), int(y)), int(rng.integers(2, 9)), color, -1)
    return pixels


def tensor(pixels):
    return torch.from_numpy(pixels.copy())[None]


def drift(pixels, dx=5, dy=-3):
    matrix = np.float64([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    return cv2.warpPerspective(pixels, matrix, pixels.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)


def test_shift_and_local_edit_are_corrected_without_replacing_edit():
    original = scene()
    intended = original.copy()
    intended[96:144, 96:144] = (0.95, 0.05, 0.1)
    shifted = drift(intended)
    source, edited = tensor(original), tensor(shifted)
    before = source.clone(), edited.clone()
    result, report = fix_pixel_drift(source, edited)
    assert "corrected" in report
    # Compare stable content and also ensure the intended replacement survives.
    region = np.s_[16:80, 16:240]
    error_before = np.abs(shifted[region] - original[region]).mean()
    error_after = np.abs(result[0].numpy()[region] - original[region]).mean()
    assert error_after < error_before * 0.25
    np.testing.assert_allclose(result[0, 108:132, 108:132], intended[108:132, 108:132], atol=0.02)
    assert torch.equal(source, before[0]) and torch.equal(edited, before[1])


@pytest.mark.parametrize("seed", [8, 12, 97])
def test_unrelated_images_pass_through_exactly(seed):
    edited = tensor(scene(seed))
    output, info = fix_pixel_drift(tensor(scene()), edited)
    assert torch.equal(output, edited)
    assert "corrected" not in info


def test_shared_logo_is_not_enough_to_align_different_pictures():
    original = scene()
    unrelated = scene(87)
    unrelated[10:100, 10:100] = original[10:100, 10:100]
    edited = tensor(drift(unrelated))
    output, info = fix_pixel_drift(tensor(original), edited)
    assert torch.equal(output, edited)
    assert "corrected" not in info


def test_shared_distributed_patches_still_require_whole_image_structure():
    original, unrelated = scene(), scene(67)
    for y in (12, 192):
        for x in (12, 192):
            unrelated[y:y + 40, x:x + 40] = original[y:y + 40, x:x + 40]
    edited = tensor(drift(unrelated))
    output, info = fix_pixel_drift(tensor(original), edited)
    assert torch.equal(output, edited)
    assert "not enough shared image structure" in info


@pytest.mark.parametrize("size", [16, 128])
def test_featureless_or_tiny_images_are_exact_noops(size):
    source = torch.full((1, size, size, 3), 0.2)
    edited = torch.full_like(source, 0.7)
    output, _ = fix_pixel_drift(source, edited)
    assert torch.equal(output, edited)


def test_already_aligned_local_edit_is_exact_noop():
    source = tensor(scene())
    edited = source.clone()
    edited[:, 95:145, 95:145] = 0.9
    result, report = fix_pixel_drift(source, edited)
    assert torch.equal(result, edited)
    assert "already aligned" in report


def test_moving_foreground_does_not_realign_stable_background():
    source = tensor(scene())
    edited = source.clone()
    edited[:, 80:160, 95:175] = source[:, 80:160, 80:160]
    output, report = fix_pixel_drift(source, edited)
    assert torch.equal(output, edited)
    assert "corrected" not in report


def test_native_warp_scales_transform_back_from_bounded_analysis():
    original = cv2.resize(scene(), (1280, 1152))
    edited = drift(original, dx=14, dy=-9)
    result, report = fix_pixel_drift(tensor(original), tensor(edited))
    assert "corrected" in report and result.shape == tensor(edited).shape
    region = np.s_[40:-40, 40:-40]
    before = np.abs(edited[region] - original[region]).mean()
    after = np.abs(result[0].numpy()[region] - original[region]).mean()
    assert after < before * 0.15


def test_excessive_reframing_is_rejected():
    original = scene()
    edited = tensor(drift(original, dx=65, dy=0))
    output, info = fix_pixel_drift(tensor(original), edited)
    assert torch.equal(output, edited)
    assert "safe drift limits" in info


def test_mild_perspective_and_contrast_change_can_be_corrected():
    original = scene()
    matrix = np.float64([[1.015, 0.008, -3], [-0.004, 0.99, 4], [0.00002, -0.00003, 1]])
    modified = original * 0.75 + 0.1
    warped = cv2.warpPerspective(modified, matrix, (256, 256), borderMode=cv2.BORDER_REPLICATE)
    result, info = fix_pixel_drift(tensor(original), tensor(warped))
    assert "corrected" in info
    # Two resampling steps lose some texture, but geometry should improve.
    after = np.abs(result[0, 16:-16, 16:-16].numpy() - modified[16:-16, 16:-16]).mean()
    before = np.abs(warped[16:-16, 16:-16] - modified[16:-16, 16:-16]).mean()
    assert after < before * 0.5


def test_source_at_different_resolution_and_mixed_batch_preserve_edit_shape():
    original = scene()
    source = tensor(cv2.resize(original, (384, 384)))
    corrected_candidate = tensor(drift(original))
    unrelated = tensor(scene(57))
    edited = torch.cat((corrected_candidate, unrelated))
    output, info = fix_pixel_drift(source, edited)
    assert output.shape == edited.shape
    assert "Image 1: corrected" in info
    assert "Image 2: corrected" not in info
    assert torch.equal(output[1], edited[1])


def test_singleton_edit_broadcasts_without_truncating_source_batch():
    edited = tensor(scene())
    sources = torch.cat((edited, tensor(scene(91))))
    output, report = fix_pixel_drift(sources, edited)
    assert output.shape[0] == 2 and len(report.splitlines()) == 2
    assert torch.equal(output, edited.expand(2, -1, -1, -1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_warp_retains_rgba_dtype_device_and_alpha(dtype, mode):
    original = scene()
    rgba = np.concatenate((original, np.full_like(original[..., :1], 0.65)), axis=-1)
    edited = tensor(drift(rgba)).to(dtype)
    output, report = fix_pixel_drift(tensor(original).to(dtype), edited, mode=mode)
    assert "corrected" in report
    assert output.dtype == dtype and output.device == edited.device
    assert output.shape == edited.shape and torch.isfinite(output).all()
    torch.testing.assert_close(output[..., 3], edited[..., 3])


@pytest.mark.parametrize("matrix", [
    np.zeros((3, 3)), np.full((3, 3), np.nan),
    np.float64([[-1, 0, 255], [0, 1, 0], [0, 0, 1]]),
    np.float64([[1, 0, 0], [0, 1, 0], [-0.01, 0, 1]]),
    np.diag([0.1, 0.1, 1]),
])
def test_invalid_transform_geometry_is_rejected(matrix):
    assert not _safe_transform(matrix, 256, 256, 0.15, cv2)


@pytest.mark.parametrize("kwargs,match", [
    ({"min_similarity": 0.0}, "min_similarity"),
    ({"min_similarity": float("nan")}, "min_similarity"),
    ({"max_drift": 0.9}, "max_drift"),
])
def test_invalid_settings_raise_actionable_errors(kwargs, match):
    image = tensor(scene())
    with pytest.raises(ValueError, match=match):
        fix_pixel_drift(image, image, **kwargs)


def test_invalid_images_and_batch_sizes_are_not_silently_ignored():
    image = tensor(scene())
    with pytest.raises(ValueError, match="batch sizes"):
        fix_pixel_drift(image.expand(2, -1, -1, -1), image.expand(3, -1, -1, -1))
    with pytest.raises(ValueError, match="shape"):
        fix_pixel_drift(image[0], image)
    with pytest.raises(ValueError, match="finite"):
        fix_pixel_drift(image, image * float("nan"))


def test_missing_opencv_does_not_break_import_but_execution_explains_dependency(monkeypatch):
    image = tensor(scene())
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(RuntimeError, match="requires OpenCV"):
        fix_pixel_drift(image, image)


def test_node_contract_and_whole_pack_registration(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("test_pixel_drift_pack", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    pack = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, pack)
    spec.loader.exec_module(pack)
    assert pack.NODE_CLASS_MAPPINGS["MAIPixelDriftFix"].__name__ == "MAIPixelDriftFix"
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["MAIPixelDriftFix"] == "mAI pixel drift fix"
    assert "MAIImagePreserveSource" in pack.NODE_CLASS_MAPPINGS
    assert MAIPixelDriftFix.CATEGORY == "mAI / Image"
    assert MAIPixelDriftFix.RETURN_TYPES == ("IMAGE", "STRING")
    assert MAIPixelDriftFix.RETURN_NAMES == ("image", "alignment_info")
    image = tensor(scene())
    output = MAIPixelDriftFix().run(image, image)
    assert torch.equal(output[0], image)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_cuda_input_retains_device_and_agrees_with_cpu(mode):
    source, edited = tensor(scene()), tensor(drift(scene()))
    expected, _ = fix_pixel_drift(source, edited, mode=mode)
    output, report = fix_pixel_drift(source.cuda(), edited.cuda(), mode=mode)
    assert "corrected" in report and output.device == edited.cuda().device
    torch.testing.assert_close(output.cpu(), expected, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("size", [128, 256, 512])
@pytest.mark.parametrize("shift", [(1, -1), (3, 2), (0.5, -0.75)])
def test_auto_detects_small_shifts_and_keeps_local_edits(size, shift):
    original = scene(size=size)
    intended = original.copy()
    intended[size // 3:size // 2, size // 3:size // 2] = (0.9, 0.1, 0.2)
    edited = drift(intended, *shift)
    output, report = MAIPixelDriftFix().run(tensor(original), tensor(edited), mode="auto")
    assert "corrected" in report and "model=translation" in report
    region = np.s_[8:size // 4, 8:-8]
    before = np.abs(edited[region] - original[region]).mean()
    after = np.abs(output[0].numpy()[region] - original[region]).mean()
    assert after < before * 0.7
    center = 5 * size // 12
    np.testing.assert_allclose(output[0, center, center].numpy(), (0.9, 0.1, 0.2), atol=0.03)


@pytest.mark.parametrize("kind", ["unrelated", "logo", "patches", "flat", "aligned", "large_shift", "moving_object"])
def test_auto_skips_unreliable_or_already_aligned_pairs_exactly(kind):
    original = scene()
    edited = scene(78)
    if kind == "logo":
        edited[10:100, 10:100] = original[10:100, 10:100]
        edited = drift(edited)
    elif kind == "patches":
        for y in (12, 192):
            for x in (12, 192):
                edited[y:y + 40, x:x + 40] = original[y:y + 40, x:x + 40]
        edited = drift(edited)
    elif kind == "flat":
        original[:] = 0.2
        edited[:] = 0.7
    elif kind in ("aligned", "moving_object"):
        edited = original.copy()
        if kind == "moving_object":
            edited[80:160, 95:175] = original[80:160, 80:160]
        else:
            edited[95:145, 95:145] = 0.9
    elif kind == "large_shift":
        edited = drift(original, dx=30, dy=0)
    candidate = tensor(edited)
    output, report = fix_pixel_drift(tensor(original), candidate, mode="auto")
    assert torch.equal(output, candidate)
    assert "corrected" not in report


def test_auto_uses_perspective_only_when_uniform_shift_cannot_explain_motion():
    original = scene()
    matrix = np.float64([[1.015, 0.008, -3], [-0.004, 0.99, 4], [0.00002, -0.00003, 1]])
    edited = cv2.warpPerspective(original, matrix, (256, 256), borderMode=cv2.BORDER_REPLICATE)
    output, report = fix_pixel_drift(tensor(original), tensor(edited), mode="auto")
    assert "corrected" in report and "model=perspective" in report
    region = np.s_[16:-16, 16:-16]
    assert np.abs(output[0].numpy()[region] - original[region]).mean() < np.abs(edited[region] - original[region]).mean() * 0.5


def test_auto_ignores_manual_controls_and_legacy_calls_keep_manual_behavior():
    source, edited = tensor(scene()), tensor(drift(scene()))
    automatic = fix_pixel_drift(source, edited, mode="auto")
    ignored = fix_pixel_drift(source, edited, min_similarity=float("nan"), max_drift=0, mode="auto")
    assert torch.equal(automatic[0], ignored[0]) and automatic[1] == ignored[1]
    manual = fix_pixel_drift(source, edited, mode="manual")
    legacy = MAIPixelDriftFix().run(source, edited)
    assert torch.equal(manual[0], legacy[0]) and manual[1] == legacy[1]
    assert MAIPixelDriftFix.INPUT_TYPES()["optional"]["mode"][1]["default"] == "auto"
    with pytest.raises(ValueError, match="Unknown pixel drift mode"):
        fix_pixel_drift(source, edited, mode="unknown")


def test_auto_high_resolution_mixed_batch_and_color_change():
    original = cv2.resize(scene(), (1280, 1152))
    edited = tensor(drift(original * 0.8 + 0.1, dx=10, dy=-6))
    unrelated = tensor(cv2.resize(scene(79), (1280, 1152)))
    batch = torch.cat((edited, unrelated))
    output, report = fix_pixel_drift(tensor(original), batch, mode="auto")
    assert "Image 1: corrected" in report and "model=translation" in report
    assert "Image 2: corrected" not in report and torch.equal(output[1], batch[1])
    expected = original * 0.8 + 0.1
    assert np.abs(output[0, 40:-40, 40:-40].numpy() - expected[40:-40, 40:-40]).mean() < 0.02


def test_auto_frontend_visibility_and_saved_workflows():
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node.js unavailable for frontend tests")
    script = Path(__file__).with_name("pixel_drift_fix_frontend.mjs")
    result = subprocess.run([executable, str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
