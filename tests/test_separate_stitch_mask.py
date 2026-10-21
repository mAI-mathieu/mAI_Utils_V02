import importlib
import importlib.util
import sys
import types
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional

from utils.separate_stitch_mask import (
    broadcast_inputs,
    crop_mask_with_stitcher_geometry,
    extend_mask,
    full_crop_feather_mask,
    feather_mask,
    image_bounds_in_crop,
    preserve_image_edges,
    preresize_dimensions,
    stack_stitcher_masks,
)
from utils import cropandstitch_dependency


class _TestProcessor:
    def rescale_m(self, samples, width, height, algorithm):
        mode = "nearest" if algorithm == "nearest" else "bilinear"
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        return torch_functional.interpolate(
            samples.unsqueeze(1), size=(height, width), mode=mode, **kwargs
        ).squeeze(1)

    def blur_m(self, samples, pixels):
        return samples


@pytest.fixture
def cropandstitch_classes(monkeypatch):
    """Load the installed upstream CPU code with minimal ComfyUI runtime stubs."""

    comfy_package = types.ModuleType("comfy")
    comfy_package.__path__ = []
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_management = types.ModuleType("comfy.model_management")
    comfy_management.get_torch_device = lambda: torch.device("cpu")
    comfy_package.utils = comfy_utils
    comfy_package.model_management = comfy_management
    comfy_nodes = types.ModuleType("nodes")
    comfy_nodes.MAX_RESOLUTION = 16384

    scipy_package = types.ModuleType("scipy")
    scipy_package.__path__ = []
    scipy_ndimage = types.ModuleType("scipy.ndimage")
    def gaussian_filter(array, sigma):
        radius = max(1, int(4.0 * sigma + 0.5))
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel_1d /= kernel_1d.sum()
        kernel = kernel_1d[:, None] * kernel_1d[None, :]
        tensor = torch.from_numpy(array).float()[None, None]
        return torch_functional.conv2d(tensor, kernel[None, None], padding=radius)[0, 0].numpy()

    scipy_ndimage.gaussian_filter = gaussian_filter
    def grey_dilation(array, footprint):
        kernel_height, kernel_width = footprint.shape
        tensor = torch.from_numpy(array).float()[None, None]
        result = torch_functional.max_pool2d(
            tensor,
            kernel_size=(kernel_height, kernel_width),
            stride=1,
            padding=(kernel_height // 2, kernel_width // 2),
        )
        return result[0, 0, : array.shape[0], : array.shape[1]].numpy()

    scipy_ndimage.grey_dilation = grey_dilation
    scipy_ndimage.binary_closing = lambda array, structure, border_value: array
    scipy_ndimage.binary_fill_holes = lambda array: array
    scipy_package.ndimage = scipy_ndimage

    for name, module in {
        "comfy": comfy_package,
        "comfy.utils": comfy_utils,
        "comfy.model_management": comfy_management,
        "nodes": comfy_nodes,
        "scipy": scipy_package,
        "scipy.ndimage": scipy_ndimage,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    repo_dir = Path(__file__).resolve().parents[1]
    upstream_file = repo_dir.parent / "comfyui-inpaint-cropandstitch" / "inpaint_cropandstitch.py"
    spec = importlib.util.spec_from_file_location("test_installed_cropandstitch", upstream_file)
    upstream = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, upstream)
    spec.loader.exec_module(upstream)

    package_name = "test_mai_crop_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(repo_dir)]
    nodes_package = types.ModuleType(f"{package_name}.nodes")
    nodes_package.__path__ = [str(repo_dir / "nodes")]
    utils_package = types.ModuleType(f"{package_name}.utils")
    utils_package.__path__ = [str(repo_dir / "utils")]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}.nodes", nodes_package)
    monkeypatch.setitem(sys.modules, f"{package_name}.utils", utils_package)

    custom_module = importlib.import_module(
        f"{package_name}.nodes.inpaint_crop_separate_stitch_mask"
    )
    yield custom_module.MAIInpaintCropSeparateStitchMask, upstream
    sys.modules.pop("_mai_inpaint_cropandstitch_dependency", None)


def _crop_inputs(image, mask):
    return {
        "image": image,
        "downscale_algorithm": "bilinear",
        "upscale_algorithm": "bicubic",
        "preresize": False,
        "preresize_mode": "ensure minimum resolution",
        "preresize_min_width": 64,
        "preresize_min_height": 64,
        "preresize_max_width": 16384,
        "preresize_max_height": 16384,
        "mask_fill_holes": False,
        "mask_expand_pixels": 0,
        "mask_invert": False,
        "mask_blend_pixels": 0,
        "mask_hipass_filter": 0.0,
        "extend_for_outpainting": False,
        "extend_up_factor": 1.0,
        "extend_down_factor": 1.0,
        "extend_left_factor": 1.0,
        "extend_right_factor": 1.0,
        "context_from_mask_extend_factor": 1.0,
        "output_resize_to_target_size": False,
        "output_target_width": 64,
        "output_target_height": 64,
        "output_padding": "0",
        "device_mode": "cpu (compatible)",
        "mask": mask,
    }


@pytest.mark.parametrize("device_mode", ["cpu (compatible)", "gpu (much faster)"])
@pytest.mark.parametrize("stitch_mode", ["rectangle (full crop)", "extended mask"])
def test_bounded_mode_native_pixels_and_upstream_stitch_round_trip(
    cropandstitch_classes, device_mode, stitch_mode,
):
    node_class, upstream = cropandstitch_classes
    image = torch.rand(1, 85, 110, 3)
    mask = torch.zeros(1, 85, 110)
    mask[:, 60:80, 90:105] = 1
    inputs = _crop_inputs(image, mask)
    # These saved upstream values must not resize or extend the bounded crop.
    inputs.update(preresize=True, extend_for_outpainting=True, extend_left_factor=2,
                  output_resize_to_target_size=True, output_padding="128", device_mode=device_mode)
    stitcher, crop, render, blend = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=80, max_crop_height=70, crop_multiple_of=12,
        stitch_mask_mode=stitch_mode, stitch_mask_expand_pixels=0, stitch_mask_blend_pixels=0,
        **inputs,
    )
    assert crop.shape == (1, 60, 72, 3)
    assert torch.equal(crop, image[:, 25:85, 38:110])
    assert torch.equal(render, mask[:, 25:85, 38:110])
    assert torch.equal(blend[0], stitcher["cropped_mask_for_blend"][0][0])
    restored, = upstream.InpaintStitchImproved().inpaint_stitch(stitcher, crop)
    # Upstream Stitch always passes through PIL, even at unchanged dimensions.
    # Crop itself above is exact; allow upstream's existing 8-bit quantization.
    assert torch.allclose(restored, image, atol=1 / 255, rtol=0)
    outside = torch.ones_like(mask, dtype=torch.bool)
    outside[:, 25:85, 38:110] = False
    assert torch.equal(restored[outside], image[outside])
    edited, = upstream.InpaintStitchImproved().inpaint_stitch(stitcher, torch.ones_like(crop))
    expected = image.clone()
    if stitch_mode == "rectangle (full crop)":
        expected[:, 25:85, 38:110] = 1
    else:
        expected[mask.bool()] = 1
    assert torch.equal(edited, expected)


def test_bounded_mode_batch_broadcast_empty_mask_and_context(cropandstitch_classes):
    node_class, _ = cropandstitch_classes
    image = torch.zeros(1, 100, 160, 3)
    mask = torch.zeros(2, 100, 160)
    mask[0, :10, :10] = 1
    mask[1, 90:, 150:] = 1
    inputs = _crop_inputs(image, mask)
    stitcher, crop, render, _ = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        **inputs,
    )
    assert crop.shape == (2, 64, 64, 3)
    assert stitcher["cropped_to_canvas_x"] == [0, 96]
    assert stitcher["cropped_to_canvas_y"] == [0, 36]
    assert render.sum() == mask.sum()

    inputs = _crop_inputs(image, None)
    inputs["optional_context_mask"] = torch.zeros(1, 100, 160)
    inputs["optional_context_mask"][:, 90:, 150:] = 1
    stitcher, _, render, _ = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        **inputs,
    )
    assert stitcher["cropped_to_canvas_x"] == [96]
    assert render.count_nonzero() == 0
    inputs.pop("optional_context_mask")
    stitcher, _, render, _ = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        **inputs,
    )
    assert stitcher["cropped_to_canvas_x"] == [48]
    assert stitcher["cropped_to_canvas_y"] == [18]
    assert render.count_nonzero() == 0


def test_bounded_mode_includes_expanded_stitch_mask_and_downscales_oversize(cropandstitch_classes):
    node_class, _ = cropandstitch_classes
    image = torch.zeros(1, 128, 128, 3)
    mask = torch.zeros(1, 128, 128)
    mask[:, 62:66, 62:66] = 1
    inputs = _crop_inputs(image, mask)
    _, _, render, blend = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        stitch_mask_mode="extended mask", stitch_mask_expand_pixels=32, stitch_mask_blend_pixels=8,
        **inputs,
    )
    assert render.count_nonzero() == 16
    assert blend.count_nonzero() > render.count_nonzero()
    assert (blend[:, 0] == 0).all()
    inputs["mask"] = torch.ones_like(mask)
    stitcher, crop, render, blend = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        stitch_mask_blend_pixels=0, **inputs,
    )
    assert crop.shape == (1, 64, 64, 3)
    assert (render == 1).all() and (blend == 1).all()
    assert stitcher["cropped_to_canvas_w"] == [128]
    assert stitcher["cropped_to_canvas_h"] == [128]


@pytest.mark.parametrize("stitch_mode", ["rectangle (full crop)", "extended mask"])
@pytest.mark.parametrize("device_mode", ["cpu (compatible)", "gpu (much faster)"])
def test_reported_oversize_mask_resizes_both_masks_and_stitches_to_source(
    cropandstitch_classes, stitch_mode, device_mode,
):
    node_class, upstream = cropandstitch_classes
    image = torch.zeros(1, 1450, 1400, 3)
    mask = torch.zeros(1, 1450, 1400)
    mask[:, 50:1338, 100:1316] = 1
    inputs = _crop_inputs(image, mask)
    inputs["device_mode"] = device_mode
    stitcher, crop, render, blend = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=1024, max_crop_height=1024, crop_multiple_of=64,
        stitch_mask_mode=stitch_mode, stitch_mask_expand_pixels=0, stitch_mask_blend_pixels=0,
        **inputs,
    )
    assert crop.shape == (1, 1024, 1024, 3)
    assert render.shape == blend.shape == (1, 1024, 1024)
    x, y = stitcher["cropped_to_canvas_x"][0], stitcher["cropped_to_canvas_y"][0]
    w, h = stitcher["cropped_to_canvas_w"][0], stitcher["cropped_to_canvas_h"][0]
    assert (x, y, w, h) == (64, 50, 1288, 1288)
    reference_mask = upstream.CPUProcessorLogic().rescale_m(
        mask[:, y:y+h, x:x+w], 1024, 1024, "bilinear",
    )
    assert torch.equal(render, reference_mask)
    assert torch.equal(blend[0], stitcher["cropped_mask_for_blend"][0][0])
    if stitch_mode == "extended mask":
        assert torch.equal(blend, reference_mask)
    edited, = upstream.InpaintStitchImproved().inpaint_stitch(stitcher, torch.ones_like(crop))
    assert edited.shape == image.shape
    assert (edited[:, 60:1328, 110:1306] == 1).all()
    outside = torch.ones_like(mask, dtype=torch.bool)
    outside[:, y:y+h, x:x+w] = False
    assert torch.equal(edited[outside], image[outside])


def test_oversize_batch_context_and_masks_share_output_dimensions(cropandstitch_classes):
    node_class, upstream = cropandstitch_classes
    image = torch.zeros(1, 150, 200, 3)
    mask = torch.zeros(2, 150, 200)
    mask[0, :10, :10] = 1
    mask[1, 60:, 100:] = 1
    context = torch.zeros_like(mask)
    context[0, :20, :80] = 1
    inputs = _crop_inputs(image, mask)
    inputs["optional_context_mask"] = context
    stitcher, crop, render, blend = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        stitch_mask_blend_pixels=0, **inputs,
    )
    assert crop.shape == (2, 64, 64, 3)
    assert render.shape == blend.shape == (2, 64, 64)
    assert stitcher["cropped_to_canvas_x"] == [0, 100]
    assert stitcher["cropped_to_canvas_y"] == [0, 50]
    assert stitcher["cropped_to_canvas_w"] == [100, 100]
    restored, = upstream.InpaintStitchImproved().inpaint_stitch(stitcher, crop)
    assert restored.shape == (2, 150, 200, 3)
    assert restored.count_nonzero() == 0


def test_bounded_mode_accepts_load_image_empty_placeholder_masks(cropandstitch_classes):
    node_class, _ = cropandstitch_classes
    inputs = _crop_inputs(torch.zeros(1, 100, 160, 3), torch.zeros(1, 64, 64))
    inputs["optional_context_mask"] = torch.zeros(64, 64)
    _, crop, render, _ = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        **inputs,
    )
    assert crop.shape == (1, 64, 64, 3)
    assert render.count_nonzero() == 0


def test_new_widgets_are_appended_and_existing_sockets_stay_stable(cropandstitch_classes):
    node_class, upstream = cropandstitch_classes
    inputs = node_class.INPUT_TYPES()
    existing = list(upstream.InpaintCropImproved.INPUT_TYPES()["required"])
    existing += ["stitch_mask_mode", "stitch_mask_expand_pixels", "stitch_mask_blend_pixels"]
    assert list(inputs["required"])[:len(existing)] == existing
    assert inputs["required"]["use_mask_max_size"][1]["default"] is False
    assert node_class.RETURN_NAMES == ("stitcher", "cropped_image", "cropped_mask", "cropped_stitch_mask")
    assert inputs["optional"] == upstream.InpaintCropImproved.INPUT_TYPES()["optional"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js unavailable")
def test_inpaint_crop_frontend_visibility_and_saved_values():
    result = subprocess.run(
        [shutil.which("node"), str(Path(__file__).with_name("inpaint_crop_frontend.mjs"))],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_broadcasts_single_image_and_masks_to_stitch_batch():
    image = torch.zeros((1, 8, 10, 3))
    render_mask = torch.zeros((1, 8, 10))
    stitch_mask = torch.zeros((4, 8, 10))

    image, render_mask, context_mask, stitch_mask = broadcast_inputs(
        image, render_mask, None, stitch_mask
    )

    assert image.shape == (4, 8, 10, 3)
    assert render_mask.shape == context_mask.shape == stitch_mask.shape == (4, 8, 10)


def test_rejects_spatially_misaligned_stitch_mask():
    image = torch.zeros((1, 8, 10, 3))
    stitch_mask = torch.zeros((1, 7, 10))

    try:
        broadcast_inputs(image, None, None, stitch_mask)
    except ValueError as exc:
        assert "stitch_mask dimensions must match image dimensions" in str(exc)
    else:
        raise AssertionError("Expected a spatial validation error")


def test_outpainting_extension_preserves_mask_position_and_uses_zero_padding():
    mask = torch.zeros((1, 4, 5))
    mask[:, 1:3, 2:4] = 1.0

    result = extend_mask(mask, 1.5, 1.0, 1.5, 1.0)

    assert result.shape == (1, 6, 7)
    assert torch.equal(result[:, 3:5, 4:6], torch.ones((1, 2, 2)))
    assert torch.count_nonzero(result) == 4


def test_preresize_dimensions_match_upstream_rounding_rules():
    assert preresize_dimensions(2048, 1365, "ensure maximum resolution", 0, 0, 1024, 1024) == (
        1024,
        682,
        "nearest",
    )
    assert preresize_dimensions(500, 333, "ensure minimum resolution", 1024, 1024, 8192, 8192) == (
        1538,
        1024,
        "bilinear",
    )


def test_canvas_geometry_keeps_render_and_larger_stitch_regions_aligned():
    stitch_mask = torch.zeros((1, 10, 12))
    stitch_mask[:, 2:8, 2:10] = 1.0
    geometry = {
        "canvas_shape": (14, 16),
        "canvas_to_orig_x": 2,
        "canvas_to_orig_y": 2,
        "canvas_to_orig_w": 12,
        "canvas_to_orig_h": 10,
        "cropped_to_canvas_x": 1,
        "cropped_to_canvas_y": 1,
        "cropped_to_canvas_w": 14,
        "cropped_to_canvas_h": 12,
    }

    result = crop_mask_with_stitcher_geometry(
        stitch_mask,
        geometry,
        12,
        14,
        _TestProcessor(),
        "bilinear",
        "bicubic",
        0,
    )

    expected = torch.zeros((1, 12, 14))
    expected[:, 3:9, 3:11] = 1.0
    assert torch.equal(result, expected)


def test_stacked_preview_exactly_matches_stitcher_masks():
    masks = [torch.rand((1, 6, 7)), torch.rand((1, 6, 7))]
    stitcher = {"cropped_mask_for_blend": masks}

    preview = stack_stitcher_masks(stitcher)

    assert torch.equal(preview[0], masks[0][0])
    assert torch.equal(preview[1], masks[1][0])


def test_full_crop_feather_reaches_exact_black_with_unlimited_setting():
    mask = full_crop_feather_mask(
        1, 32, 48, 10000, torch.device("cpu"), torch.float32
    )

    assert torch.all(mask[:, 0, :] == 0.0)
    assert torch.all(mask[:, -1, :] == 0.0)
    assert torch.all(mask[:, :, 0] == 0.0)
    assert torch.all(mask[:, :, -1] == 0.0)
    assert mask[0, 15, 23] == 1.0
    assert torch.any((mask > 0.0) & (mask < 1.0))


def test_full_crop_feather_has_only_one_exact_black_perimeter():
    height, width = 128, 160
    mask = full_crop_feather_mask(
        1, height, width, 64, torch.device("cpu"), torch.float32
    )

    expected_perimeter_pixels = 2 * height + 2 * width - 4
    assert torch.count_nonzero(mask == 0.0) == expected_perimeter_pixels
    assert torch.all(mask[:, 1, 1:-1] > 0.0)
    assert torch.all(mask[:, -2, 1:-1] > 0.0)
    assert torch.all(mask[:, 1:-1, 1] > 0.0)
    assert torch.all(mask[:, 1:-1, -2] > 0.0)


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_feather_preserves_mask_values_on_picture_edges_only(edge):
    mask = torch.zeros(1, 32, 32)
    mask[:, 4:28, 4:28] = 1
    # Include a soft segment: protecting the border must not force it to white.
    mask[:, 10:14] *= 0.4
    bounds = {
        "left": (0, -10, 50, 50), "right": (-10, -10, 32, 50),
        "top": (-10, 0, 50, 50), "bottom": (-10, -10, 50, 32),
    }[edge]
    if edge == "left":
        mask[:, :, :8] = mask[:, :, 4:5].clone()
    elif edge == "right":
        mask[:, :, -8:] = mask[:, :, -5:-4].clone()
    elif edge == "top":
        mask[:, :8, :] = mask[:, 4:5, :].clone()
    else:
        mask[:, -8:, :] = mask[:, -5:-4, :].clone()
    result = feather_mask(mask, 8, _TestProcessor(), bounds)
    selection = {"left": (slice(None), slice(None), 0),
                 "right": (slice(None), slice(None), -1),
                 "top": (slice(None), 0, slice(None)),
                 "bottom": (slice(None), -1, slice(None))}[edge]
    assert torch.equal(result[selection], mask[selection])
    assert result[0, 0 if edge != "top" else -1, 16] == 0


def test_picture_border_inside_padding_retains_values_and_empty_region():
    mask = torch.zeros(1, 40, 48)
    mask[:, 6:30, 8:40] = 1
    mask[:, 15:20, 8] = 0.25
    bounds = (8, 6, 40, 30)
    result = preserve_image_edges(mask, torch.zeros_like(mask), bounds, 8)
    assert torch.equal(result[:, 6, 8:40], mask[:, 6, 8:40])
    assert torch.equal(result[:, 29, 8:40], mask[:, 29, 8:40])
    assert torch.equal(result[:, 6:30, 8], mask[:, 6:30, 8])
    assert torch.equal(result[:, 6:30, 39], mask[:, 6:30, 39])
    assert result[:, :6].count_nonzero() == 0


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("stitch_mode", ["rectangle (full crop)", "extended mask"])
@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_stitch_mask_does_not_fade_at_original_picture_edge(
    cropandstitch_classes, bounded, stitch_mode, edge,
):
    node_class, upstream = cropandstitch_classes
    image = torch.zeros(1, 128, 128, 3)
    mask = torch.zeros(1, 128, 128)
    if edge == "left":
        mask[:, 48:80, :16] = 1
    elif edge == "right":
        mask[:, 48:80, -16:] = 1
    elif edge == "top":
        mask[:, :16, 48:80] = 1
    else:
        mask[:, -16:, 48:80] = 1
    inputs = _crop_inputs(image, mask)
    if not bounded:
        inputs.update(output_resize_to_target_size=True, output_target_width=64, output_target_height=64)
    stitcher, crop, _, blend = node_class().inpaint_crop(
        use_mask_max_size=bounded, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        stitch_mask_mode=stitch_mode, stitch_mask_expand_pixels=0, stitch_mask_blend_pixels=8,
        **inputs,
    )
    geometry = node_class._geometry_for(stitcher, 0, inputs)
    bounds = image_bounds_in_crop(geometry, blend.shape[1], blend.shape[2])
    left, top, right, bottom = bounds
    if edge in ("left", "right"):
        column = int(left) if edge == "left" else int(right) - 1
        assert blend[0, blend.shape[1] // 2, column] == 1
    else:
        row = int(top) if edge == "top" else int(bottom) - 1
        assert blend[0, row, blend.shape[2] // 2] == 1
    assert torch.any((blend > 0) & (blend < 1))
    assert torch.equal(blend[0], stitcher["cropped_mask_for_blend"][0][0])
    edited, = upstream.InpaintStitchImproved().inpaint_stitch(stitcher, torch.ones_like(crop))
    row, column = {"left": (64, 0), "right": (64, 127),
                   "top": (0, 64), "bottom": (127, 64)}[edge]
    assert (edited[0, row, column] == 1).all()


@pytest.mark.parametrize("stitch_mode", ["rectangle (full crop)", "extended mask"])
@pytest.mark.parametrize("device_mode", ["cpu (compatible)", "gpu (much faster)"])
def test_full_picture_oversize_mask_has_no_frame_fade_after_resizing(
    cropandstitch_classes, stitch_mode, device_mode,
):
    node_class, upstream = cropandstitch_classes
    inputs = _crop_inputs(torch.zeros(1, 128, 128, 3), torch.ones(1, 128, 128))
    inputs["device_mode"] = device_mode
    inputs["mask_blend_pixels"] = 8
    stitcher, crop, render, blend = node_class().inpaint_crop(
        use_mask_max_size=True, max_crop_width=64, max_crop_height=64, crop_multiple_of=16,
        stitch_mask_mode=stitch_mode, stitch_mask_expand_pixels=0, stitch_mask_blend_pixels=16,
        **inputs,
    )
    assert crop.shape == (1, 64, 64, 3)
    assert (blend == 1).all()
    assert (render[:, 0] == 1).all() and (render[:, -1] == 1).all()
    assert (render[:, :, 0] == 1).all() and (render[:, :, -1] == 1).all()
    edited, = upstream.InpaintStitchImproved().inpaint_stitch(stitcher, torch.ones_like(crop))
    assert (edited == 1).all()


def test_missing_dependency_raises_a_nonfatal_classified_error(monkeypatch):
    monkeypatch.setattr(cropandstitch_dependency, "_already_loaded_module", lambda: None)
    monkeypatch.setattr(cropandstitch_dependency, "_dependency_candidates", lambda: [])

    with pytest.raises(cropandstitch_dependency.CropAndStitchDependencyError) as exc_info:
        cropandstitch_dependency.load_cropandstitch_module()

    assert exc_info.value.missing is True


def test_installed_upstream_crop_and_stitch_accepts_extended_mask(cropandstitch_classes):
    node_class, upstream = cropandstitch_classes
    image = torch.zeros((1, 64, 64, 3))
    render_mask = torch.zeros((1, 64, 64))
    render_mask[:, 28:36, 28:36] = 1.0
    stitcher, cropped_image, cropped_mask, cropped_stitch_mask = node_class().inpaint_crop(
        stitch_mask_mode="extended mask",
        stitch_mask_expand_pixels=32,
        stitch_mask_blend_pixels=0,
        **_crop_inputs(image, render_mask),
    )

    assert cropped_image.shape[1:3] == cropped_mask.shape[1:] == cropped_stitch_mask.shape[1:]
    assert torch.count_nonzero(cropped_mask) == 64
    assert torch.count_nonzero(cropped_stitch_mask) == 400
    assert torch.equal(cropped_stitch_mask[0], stitcher["cropped_mask_for_blend"][0][0])

    rendered_crop = torch.ones_like(cropped_image)
    stitched, = upstream.InpaintStitchImproved().inpaint_stitch(stitcher, rendered_crop)
    expected_mask = upstream.CPUProcessorLogic().expand_m(render_mask, 32)
    expected = expected_mask.unsqueeze(-1).expand_as(stitched)
    assert torch.equal(stitched, expected)


def test_rectangle_mode_preserves_upstream_crop_and_blends_full_crop(cropandstitch_classes):
    node_class, upstream = cropandstitch_classes
    image = torch.rand((1, 48, 64, 3))
    render_mask = torch.zeros((1, 48, 64))
    render_mask[:, 10:30, 20:40] = 1.0
    inputs = _crop_inputs(image, render_mask)

    upstream_result = upstream.InpaintCropImproved().inpaint_crop(**inputs)
    custom_result = node_class().inpaint_crop(
        stitch_mask_mode="rectangle (full crop)",
        stitch_mask_expand_pixels=32,
        stitch_mask_blend_pixels=0,
        **inputs,
    )

    assert torch.equal(custom_result[1], upstream_result[1])
    assert torch.equal(custom_result[2], upstream_result[2])
    assert torch.all(custom_result[3] == 1)
    assert torch.equal(custom_result[3][0], custom_result[0]["cropped_mask_for_blend"][0][0])


def test_whole_pack_imports_and_registers_node(cropandstitch_classes, monkeypatch):
    node_class, _ = cropandstitch_classes
    repo_dir = Path(__file__).resolve().parents[1]
    module_name = "test_mai_complete_pack"
    spec = importlib.util.spec_from_file_location(
        module_name, repo_dir / "__init__.py", submodule_search_locations=[str(repo_dir)]
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, package)
    spec.loader.exec_module(package)

    mapping_key = "MAIInpaintCropSeparateStitchMask"
    assert mapping_key in package.NODE_CLASS_MAPPINGS
    assert package.NODE_DISPLAY_NAME_MAPPINGS[mapping_key] == (
        "mAI Inpaint Crop - Separate Stitch Mask"
    )
    input_types = node_class.INPUT_TYPES()
    assert "stitch_mask_mode" in input_types["required"]
    assert "stitch_mask_expand_pixels" in input_types["required"]
    assert "stitch_mask_blend_pixels" in input_types["required"]
    assert input_types["required"]["stitch_mask_blend_pixels"][1]["max"] == 16384
    assert "stitch_mask" not in input_types["optional"]


def test_target_resize_and_different_aspect_ratio_remain_aligned(cropandstitch_classes):
    node_class, upstream = cropandstitch_classes
    image = torch.zeros((1, 1365, 2048, 3))
    render_mask = torch.zeros((1, 1365, 2048))
    render_mask[:, 580:785, 970:1070] = 1.0
    inputs = _crop_inputs(image, render_mask)
    inputs.update(
        output_resize_to_target_size=True,
        output_target_width=1024,
        output_target_height=1024,
        downscale_algorithm="nearest",
        upscale_algorithm="nearest",
    )

    stitcher, cropped_image, cropped_mask, cropped_stitch_mask = node_class().inpaint_crop(
        stitch_mask_mode="extended mask",
        stitch_mask_expand_pixels=32,
        stitch_mask_blend_pixels=0,
        **inputs,
    )

    assert cropped_image.shape == (1, 1024, 1024, 3)
    assert cropped_mask.shape == cropped_stitch_mask.shape == (1, 1024, 1024)
    assert torch.all(cropped_stitch_mask[cropped_mask > 0] == 1)
    stitched, = upstream.InpaintStitchImproved().inpaint_stitch(
        stitcher, torch.ones_like(cropped_image)
    )
    expected_mask = upstream.CPUProcessorLogic().expand_m(render_mask, 32)
    assert torch.count_nonzero(stitched[expected_mask == 0]) == 0


def test_batch_broadcast_and_context_mask_do_not_change_mask_roles(cropandstitch_classes):
    node_class, _ = cropandstitch_classes
    image = torch.zeros((4, 64, 64, 3))
    render_mask = torch.zeros((1, 64, 64))
    render_mask[:, 28:36, 28:36] = 1.0
    context_mask = torch.zeros((1, 64, 64))
    context_mask[:, 4:8, 4:8] = 1.0
    inputs = _crop_inputs(image, render_mask)
    inputs.update(
        output_resize_to_target_size=True,
        output_target_width=64,
        output_target_height=64,
        optional_context_mask=context_mask,
    )

    stitcher, cropped_image, cropped_mask, cropped_stitch_mask = node_class().inpaint_crop(
        stitch_mask_mode="extended mask",
        stitch_mask_expand_pixels=32,
        stitch_mask_blend_pixels=0,
        **inputs,
    )

    assert cropped_image.shape[0] == cropped_mask.shape[0] == cropped_stitch_mask.shape[0] == 4
    assert len(stitcher["cropped_mask_for_blend"]) == 4
    assert torch.all(cropped_stitch_mask[cropped_mask > 0] == 1)
    assert torch.count_nonzero(cropped_stitch_mask) > torch.count_nonzero(cropped_mask)


def test_edge_mask_outpainting_and_feathering(cropandstitch_classes):
    node_class, upstream = cropandstitch_classes
    image = torch.zeros((1, 64, 64, 3))
    render_mask = torch.zeros((1, 64, 64))
    render_mask[:, 0:8, 0:8] = 1.0
    inputs = _crop_inputs(image, render_mask)
    inputs.update(
        extend_for_outpainting=True,
        extend_left_factor=1.5,
    )

    stitcher, cropped_image, _, cropped_stitch_mask = node_class().inpaint_crop(
        stitch_mask_mode="extended mask",
        stitch_mask_expand_pixels=32,
        stitch_mask_blend_pixels=16,
        **inputs,
    )

    assert cropped_stitch_mask.min() >= 0.0
    assert cropped_stitch_mask.max() <= 1.0
    assert torch.any((cropped_stitch_mask > 0.0) & (cropped_stitch_mask < 1.0))
    assert torch.equal(cropped_stitch_mask[0], stitcher["cropped_mask_for_blend"][0][0])
    stitched, = upstream.InpaintStitchImproved().inpaint_stitch(
        stitcher, torch.ones_like(cropped_image)
    )
    assert stitched.shape[1:3] == (64, 96)
    assert torch.count_nonzero(stitched[:, :, :16]) == 0
    # The original picture starts at x=32 inside the extended canvas.
    assert (stitched[0, 0, 32:40] == 1).all()


@pytest.mark.parametrize("stitch_mode", ["rectangle (full crop)", "extended mask"])
def test_picture_corner_stays_white_with_preresize_and_padding(cropandstitch_classes, stitch_mode):
    node_class, upstream = cropandstitch_classes
    image = torch.zeros(1, 64, 64, 3)
    mask = torch.zeros(1, 64, 64)
    mask[:, :16, :16] = 1
    inputs = _crop_inputs(image, mask)
    inputs.update(
        preresize=True, preresize_min_width=128, preresize_min_height=128,
        output_resize_to_target_size=True, output_target_width=64, output_target_height=64,
        output_padding="32", downscale_algorithm="nearest", upscale_algorithm="nearest",
    )
    stitcher, crop, _, blend = node_class().inpaint_crop(
        stitch_mask_mode=stitch_mode, stitch_mask_expand_pixels=0, stitch_mask_blend_pixels=16,
        **inputs,
    )
    bounds = image_bounds_in_crop(node_class._geometry_for(stitcher, 0, inputs), 64, 64)
    left, top = int(bounds[0]), int(bounds[1])
    assert blend[0, top, left] == 1
    edited, = upstream.InpaintStitchImproved().inpaint_stitch(stitcher, torch.ones_like(crop))
    assert edited.shape == (1, 128, 128, 3)
    assert (edited[0, 0, 0] == 1).all()


def test_rectangle_mode_feathers_the_full_crop_boundary(cropandstitch_classes):
    node_class, _ = cropandstitch_classes
    image = torch.zeros((1, 64, 64, 3))
    render_mask = torch.zeros((1, 64, 64))
    render_mask[:, 16:48, 16:48] = 1.0

    stitcher, _, _, cropped_stitch_mask = node_class().inpaint_crop(
        stitch_mask_mode="rectangle (full crop)",
        stitch_mask_expand_pixels=32,
        stitch_mask_blend_pixels=16,
        **_crop_inputs(image, render_mask),
    )

    center = cropped_stitch_mask[0, cropped_stitch_mask.shape[1] // 2, cropped_stitch_mask.shape[2] // 2]
    corner = cropped_stitch_mask[0, 0, 0]
    assert center > corner
    assert center == 1.0
    assert torch.all(cropped_stitch_mask[:, 0, :] == 0.0)
    assert torch.all(cropped_stitch_mask[:, -1, :] == 0.0)
    assert torch.all(cropped_stitch_mask[:, :, 0] == 0.0)
    assert torch.all(cropped_stitch_mask[:, :, -1] == 0.0)
    assert torch.any((cropped_stitch_mask > 0.0) & (cropped_stitch_mask < 1.0))
    assert torch.equal(cropped_stitch_mask[0], stitcher["cropped_mask_for_blend"][0][0])
