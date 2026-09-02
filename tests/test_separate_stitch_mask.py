import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional

from utils.separate_stitch_mask import (
    broadcast_inputs,
    crop_mask_with_stitcher_geometry,
    extend_mask,
    full_crop_feather_mask,
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
