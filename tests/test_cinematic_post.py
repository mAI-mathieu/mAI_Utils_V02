import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from nodes.cinematic_post import mAI_CinematicPost
from utils.cinematic_post import (
    FLOAT_CONTROLS, PRESETS, apply_bloom, apply_chromatic_aberration,
    apply_color_density, apply_film_grain, apply_filmic_curve, apply_halation,
    apply_vignette, cinematic_post, gaussian_blur, luminance, resolve_settings,
)


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample_image(batch=1, height=24, width=32):
    generator = torch.Generator().manual_seed(42)
    return torch.rand((batch, height, width, 3), generator=generator)


@pytest.mark.parametrize("options", [{"enabled": False}, {"strength": 0}, {"preset": "Off / Neutral"}])
def test_bypass_is_exact_and_does_not_process_mask(options):
    image = sample_image()
    assert cinematic_post(image, subject_mask="unused", **options) is image


@pytest.mark.parametrize("shape", [(1, 1, 1, 3), (2, 1, 7, 3), (1, 9, 1, 3), (2, 13, 17, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_preserves_shape_dtype_range_and_input(shape, dtype):
    image = torch.linspace(0, 1, steps=math.prod(shape)).reshape(shape).to(dtype)
    before = image.clone()
    output = cinematic_post(image)
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert output.device == image.device
    assert torch.isfinite(output).all()
    assert output.min() >= 0 and output.max() <= 1
    assert torch.equal(image, before)


def test_batch_matches_individual_processing_and_grain_is_reproducible():
    image = sample_image(batch=2)
    state = torch.random.get_rng_state().clone()
    result = cinematic_post(image, grain_seed=123)
    separate = torch.cat([cinematic_post(frame[None], grain_seed=123) for frame in image])
    assert torch.equal(result, separate)
    assert torch.equal(result, cinematic_post(image, grain_seed=123))
    assert not torch.equal(result, cinematic_post(image, grain_seed=124))
    assert torch.equal(state, torch.random.get_rng_state())


def test_animation_flag_uses_repeatable_batch_seed_variation():
    image = torch.full((2, 24, 32, 3), 0.4)
    fixed = cinematic_post(image)
    varied = cinematic_post(image, grain_animation_safe=True)
    assert torch.equal(fixed[0], fixed[1])
    assert not torch.equal(varied[0], varied[1])
    assert torch.equal(varied, cinematic_post(image, grain_animation_safe=True))


def test_strength_blends_finished_result():
    image = sample_image()
    full = cinematic_post(image)
    half = cinematic_post(image, strength=0.5)
    assert torch.allclose(half, torch.lerp(image, full, 0.5), atol=1e-6)


def test_optional_and_broadcast_masks():
    image = sample_image(batch=2)
    assert torch.equal(cinematic_post(image), cinematic_post(image, subject_mask=torch.empty(0)))
    mask = torch.linspace(0, 1, 12).reshape(1, 3, 4)
    broadcast = cinematic_post(image, subject_mask=mask)
    repeated = cinematic_post(image, subject_mask=mask.expand(2, -1, -1))
    assert torch.equal(broadcast, repeated)
    assert torch.equal(broadcast, cinematic_post(image, subject_mask=mask[0]))
    assert torch.equal(broadcast, cinematic_post(image, subject_mask=mask[..., None]))
    assert torch.equal(broadcast, cinematic_post(image, subject_mask=mask[:, None]))
    with pytest.raises(ValueError, match="batch"):
        cinematic_post(image, subject_mask=torch.zeros(3, 4, 4))


def test_mask_preserves_saturation_when_requested():
    image = torch.tensor([0.65, 0.35, 0.2])[None, :, None, None].expand(1, 3, 8, 8)
    unprotected = apply_color_density(image, -0.6, 0, 0, 0, torch.zeros(1, 1, 8, 8))
    protected = apply_color_density(image, -0.6, 0, 0, 0, torch.ones(1, 1, 8, 8))
    assert torch.allclose(protected, image)
    assert (unprotected[:, 0] - unprotected[:, 2]).mean() < (protected[:, 0] - protected[:, 2]).mean()
    white_mask = cinematic_post(image.permute(0, 2, 3, 1), subject_mask=torch.ones(8, 8))
    black_mask = cinematic_post(image.permute(0, 2, 3, 1), subject_mask=torch.zeros(8, 8))
    assert not torch.equal(white_mask, black_mask)


def test_presets_and_control_trims():
    assert resolve_settings()["contrast"] == 0.15
    assert resolve_settings("Moody")["contrast"] == pytest.approx(0.22)
    assert resolve_settings("Moody", contrast=0.20)["contrast"] == pytest.approx(0.27)
    image = sample_image()
    outputs = [cinematic_post(image, preset=preset) for preset in PRESETS]
    for index, output in enumerate(outputs):
        assert torch.isfinite(output).all()
        for other in outputs[:index]:
            assert not torch.equal(output, other)
    assert torch.equal(cinematic_post(image, advanced_mode=False), cinematic_post(image, advanced_mode=True))


def test_filmic_curve_is_monotonic_and_lifts_black_with_smooth_shoulder():
    ramp = torch.linspace(0, 1, 1024)[None, None, None].expand(1, 3, 1, 1024)
    output = apply_filmic_curve(ramp, 0, 0.15, 0.03, 0.25, 0.15)
    assert (output.diff(dim=-1) >= -1e-7).all()
    assert output[0, 0, 0, 0] == pytest.approx(0.03)
    assert output[0, 0, 0, -1] < 1
    assert output.diff(dim=-1)[..., -1].mean() < output.diff(dim=-1)[..., 700].mean()


def test_density_darkens_color_without_affecting_gray():
    gray = torch.full((1, 3, 4, 4), 0.5)
    color = gray.clone()
    color[:, 0] = 0.8
    protection = torch.zeros(1, 1, 4, 4)
    assert torch.allclose(apply_color_density(gray, 0, 0.4, 0, 0, protection), gray)
    dense = apply_color_density(color, 0, 0.4, 0, 0, protection)
    assert luminance(dense).mean() < luminance(color).mean()
    assert torch.allclose(dense[:, 0] / dense[:, 1], color[:, 0] / color[:, 1])


def test_halation_is_warm_outside_highlights_and_bloom_is_neutral():
    image = torch.zeros(1, 3, 41, 41)
    image[:, :, 18:23, 18:23] = 0.95
    halo = apply_halation(image, 0.3, 0.75, 3)
    glow = apply_bloom(image, 0.3, 0.8, 6)
    assert halo[0, 0, 20, 16] > halo[0, 1, 20, 16] > halo[0, 2, 20, 16] > 0
    assert torch.equal(halo[:, :, 20, 20], image[:, :, 20, 20])
    assert torch.equal(glow[:, 0], glow[:, 1])
    assert glow[0, 0, 20, 8] > halo[0, 0, 20, 8]
    dark = torch.full_like(image, 0.3)
    assert torch.equal(apply_halation(dark, 0.5, 0.75, 3), dark)
    assert torch.equal(apply_bloom(dark, 0.5, 0.8, 6), dark)


@pytest.mark.parametrize("sigma", [0.5, 3, 20, 160])
def test_large_blurs_preserve_constant_images_including_tiny_inputs(sigma):
    for height, width in ((1, 1), (1, 19), (31, 17)):
        image = torch.full((1, 3, height, width), 0.4)
        assert torch.allclose(gaussian_blur(image, sigma), image, atol=1e-6)


def test_lens_effects_preserve_constant_color_and_darkening_is_peripheral():
    image = torch.full((1, 3, 33, 33), 0.5)
    assert torch.allclose(apply_chromatic_aberration(image, 2, 0.02, 1), image)
    vignette = apply_vignette(image, 0.1, 0.75)
    assert vignette[0, 0, 16, 16] == 0.5
    assert 0.45 <= vignette[0, 0, 0, 0] < 0.5


def test_grain_chroma_size_and_endpoint_protection():
    image = torch.full((1, 3, 64, 64), 0.5)
    mono = apply_film_grain(image, 0.1, 1, 0, 12, 1)
    chroma = apply_film_grain(image, 0.1, 1, 0.5, 12, 1)
    coarse = apply_film_grain(image, 0.1, 4, 0, 12, 1)
    assert torch.equal(mono[:, 0], mono[:, 1])
    assert not torch.equal(chroma[:, 0], chroma[:, 1])
    assert coarse.diff(dim=-1).abs().mean() < mono.diff(dim=-1).abs().mean()
    for level in (0, 1):
        endpoint = torch.full_like(image, level)
        assert torch.equal(apply_film_grain(endpoint, 0.5, 1, 1, 0, 1), endpoint)


@pytest.mark.parametrize("bound", [1, 2])
def test_all_controls_at_extremes_remain_finite(bound):
    controls = {name: spec[bound] for name, spec in FLOAT_CONTROLS.items()}
    controls["strength"] = 1
    result = cinematic_post(sample_image(), **controls)
    assert torch.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1


def test_invalid_inputs_have_clear_errors():
    with pytest.raises(ValueError, match="preset"):
        cinematic_post(sample_image(), preset="Unknown")
    with pytest.raises(ValueError, match="finite"):
        cinematic_post(sample_image(), contrast=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        cinematic_post(torch.full((1, 2, 2, 3), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        cinematic_post(sample_image(), subject_mask=torch.full((2, 2), float("inf")))
    with pytest.raises(ValueError, match="non-empty RGB"):
        cinematic_post(torch.zeros(1, 0, 2, 3))
    with pytest.raises(ValueError, match="non-empty RGB"):
        cinematic_post(torch.zeros(1, 2, 2, 4))
    with pytest.raises(TypeError, match="floating-point"):
        cinematic_post(torch.zeros(1, 2, 2, 3, dtype=torch.uint8))
    with pytest.raises(ValueError, match="grain_seed"):
        cinematic_post(sample_image(), grain_seed=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_runs_on_input_device_with_close_cpu_result():
    image = sample_image()
    cpu = cinematic_post(image, grain_strength=0)
    gpu = cinematic_post(image.cuda(), grain_strength=0)
    assert gpu.device.type == "cuda"
    assert torch.allclose(cpu, gpu.cpu(), atol=2e-5)
    grain = cinematic_post(image.cuda(), grain_seed=7)
    assert torch.equal(grain, cinematic_post(image.cuda(), grain_seed=7))


def test_node_contract_and_pack_registration(monkeypatch):
    schema = mAI_CinematicPost.INPUT_TYPES()
    assert schema["required"]["image"] == ("IMAGE",)
    assert schema["optional"] == {"subject_mask": ("MASK",)}
    defaults = {name: spec[1]["default"] for name, spec in schema["required"].items() if name != "image"}
    image = sample_image()
    assert torch.equal(mAI_CinematicPost().process(image, **defaults)[0], cinematic_post(image))
    root = Path(__file__).resolve().parents[1]
    name = "test_mai_cinematic_pack"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["mAI_CinematicPost"].__name__ == "mAI_CinematicPost"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["mAI_CinematicPost"] == "mAI Cinematic Post"
    assert mAI_CinematicPost.RETURN_TYPES == ("IMAGE",)
    assert mAI_CinematicPost.RETURN_NAMES == ("image",)
    assert mAI_CinematicPost.CATEGORY == "mAI / Image"
    assert "MAIBackgroundLightingMatch" in package.NODE_CLASS_MAPPINGS
    assert "MAIVideoLoader" in package.NODE_CLASS_MAPPINGS
