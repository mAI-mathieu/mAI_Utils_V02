import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from nodes.fast_gpu_resize import MAIFastGPUResize
from nodes.stitch import MAIStitch
from utils.resize_kernels import RESIZE_MODES
from utils.resize_stitch import create_resize_stitcher, validate_resize_stitcher


@pytest.fixture(autouse=True, scope="module")
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("mode", ["pad_to_multiple", "pad_to_multiple_edge"])
@pytest.mark.parametrize("shape,multiple", [
    ((3, 9, 19, 4), 8),  # odd differences: left=2, top=3, right=3, bottom=4
    ((2, 8, 16, 3), 8),  # already aligned
    ((1, 1, 1, 1), 16),  # padding larger than source
    ((2, 7, 11, 3), 1),  # disabled padding
])
@pytest.mark.parametrize("chunk_size", [0, 2])
def test_padding_round_trip_preserves_exact_pixels_and_masks(mode, shape, multiple, chunk_size):
    # Float64 and out-of-range values expose unwanted resampling, casting or clamping.
    image = torch.linspace(-1.23456789, 2.34567891, torch.Size(shape).numel(),
                           dtype=torch.float64).reshape(shape)
    mask = image[..., 0].float()
    padded, _, _, _, padded_mask, stitcher = MAIFastGPUResize().run(
        image, resize_mode=mode, multiple_of=multiple, precision="fp16",
        chunk_size=chunk_size, mask=mask,
    )
    before_image, before_mask = padded.clone(), padded_mask.clone()
    result, width, height, output_mask = MAIStitch().run(padded, stitcher, padded_mask)
    assert (width, height) == (shape[2], shape[1])
    assert torch.equal(result, image) and result.dtype == image.dtype
    assert torch.equal(output_mask, mask) and output_mask.dtype == mask.dtype
    assert result.device == padded.device and output_mask.device == result.device
    assert torch.equal(padded, before_image) and torch.equal(padded_mask, before_mask)


def test_saved_offsets_recut_processed_batch_without_recentering_or_resampling(monkeypatch):
    padded, _, _, _, _, stitcher = MAIFastGPUResize().run(
        torch.zeros(1, 9, 19, 3), resize_mode="pad_to_multiple", multiple_of=8,
    )
    assert (stitcher["x"], stitcher["y"], stitcher["width"], stitcher["height"]) == (2, 3, 19, 9)
    # A generated batch can have a different frame/channel count from the source.
    processed = torch.arange(4 * 16 * 24 * 4).double().reshape(4, 16, 24, 4)
    single_mask = torch.arange(16 * 24).float().reshape(16, 24)

    def forbidden(*args, **kwargs):
        raise AssertionError("Stitch must not resize")

    monkeypatch.setattr("utils.resize_kernels.resize_native", forbidden)
    result, width, height, mask = MAIStitch().run(processed, stitcher, single_mask)
    assert (width, height) == (19, 9)
    assert torch.equal(result, processed[:, 3:12, 2:21])
    assert torch.equal(mask, single_mask[3:12, 2:21].expand(4, -1, -1))


@pytest.mark.parametrize("target", [(24, 25), (33, 16)])
def test_fit_padding_recuts_resized_content_at_saved_scale(target):
    image = torch.arange(2 * 9 * 19 * 3).float().reshape(2, 9, 19, 3)
    padded, _, _, _, _, stitcher = MAIFastGPUResize().run(
        image, *target, resize_mode="keep_aspect_fit", method="nearest",
    )
    resized = MAIFastGPUResize().run(
        image, *target, resize_mode="keep_aspect", method="nearest",
    )[0]
    result, width, height, mask = MAIStitch().run(padded, stitcher)
    assert torch.equal(result, resized)
    assert (width, height) == (resized.shape[2], resized.shape[1])
    assert mask.shape == result.shape[:3] and not mask.any()


@pytest.mark.parametrize("mode", [m for m in RESIZE_MODES if m not in (
    "keep_aspect_fit", "pad_to_multiple", "pad_to_multiple_edge",
)])
def test_nonpadding_modes_use_full_output_rectangle(mode):
    image, width, height, _, _, stitcher = MAIFastGPUResize().run(
        torch.rand(2, 9, 19, 3), 8, 6, resize_mode=mode,
    )
    result, w, h, mask = MAIStitch().run(image, stitcher)
    assert torch.equal(result, image) and (w, h) == (width, height)
    assert mask.shape == (2, h, w) and not mask.any()


@pytest.mark.parametrize("changes,message", [
    ({"version": 2}, "version 1"),
    ({"x": -1}, "x must be an integer"),
    ({"y": True}, "y must be an integer"),
    ({"width": 0}, "width must be an integer"),
    ({"height": 1.5}, "height must be an integer"),
    ({"canvas_width": None}, "canvas_width must be an integer"),
    ({"x": 20}, "exceeds its saved canvas"),
    ({"height": 24}, "exceeds its saved canvas"),
])
def test_invalid_metadata_has_clear_errors(changes, message):
    stitcher = create_resize_stitcher(19, 9, 1, 1, "pad_to_multiple", 8)
    stitcher.update(changes)
    with pytest.raises(ValueError, match=message):
        validate_resize_stitcher(stitcher, 24, 16)


@pytest.mark.parametrize("stitcher", [None, (), {}, {"version": 1}])
def test_missing_or_unrelated_metadata_is_rejected(stitcher):
    with pytest.raises(ValueError, match="stitcher"):
        MAIStitch().run(torch.zeros(1, 16, 24, 3), stitcher)


def test_changed_canvas_is_rejected_instead_of_silently_cropping_wrong_position():
    stitcher = create_resize_stitcher(19, 9, 1, 1, "pad_to_multiple", 8)
    with pytest.raises(ValueError, match="does not match stitcher canvas"):
        MAIStitch().run(torch.zeros(1, 32, 48, 3), stitcher)


@pytest.mark.parametrize("image", [None, torch.zeros(16, 24, 3), torch.zeros(0, 16, 24, 3),
                                  torch.zeros(1, 16, 24, 3, dtype=torch.uint8)])
def test_invalid_images_are_rejected(image):
    stitcher = create_resize_stitcher(19, 9, 1, 1, "pad_to_multiple", 8)
    with pytest.raises(ValueError, match="image must"):
        MAIStitch().run(image, stitcher)


@pytest.mark.parametrize("mask,message", [
    (torch.zeros(1, 9, 19), "must match the source image"),
    (torch.zeros(3, 16, 24), "match the image batch size"),
])
def test_stitch_mask_requires_processed_canvas_and_compatible_batch(mask, message):
    stitcher = create_resize_stitcher(19, 9, 1, 1, "pad_to_multiple", 8)
    with pytest.raises(ValueError, match=message):
        MAIStitch().run(torch.zeros(2, 16, 24, 3), stitcher, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_stitch_keeps_cuda_device_and_precision(dtype):
    image = torch.rand(2, 9, 19, 3, device="cuda", dtype=dtype)
    padded, _, _, _, _, stitcher = MAIFastGPUResize().run(
        image, resize_mode="pad_to_multiple_edge", multiple_of=8,
    )
    result, _, _, mask = MAIStitch().run(padded, stitcher, torch.ones(1, 16, 24))
    assert torch.equal(result, image) and result.dtype == dtype
    assert result.device == image.device and mask.device == image.device
    assert mask.dtype == torch.float32 and mask.eq(1).all()


def test_node_registration_and_socket_contract(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "test_mai_stitch_pack", root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIStitch"].__name__ == "MAIStitch"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIStitch"] == "mAI stitch"
    assert "MAIFastGPUResize" in package.NODE_CLASS_MAPPINGS
    assert "MAIMaskSmartStitch" in package.NODE_CLASS_MAPPINGS
    assert MAIStitch.CATEGORY == "mAI / Image"
    assert MAIStitch.RETURN_TYPES == ("IMAGE", "INT", "INT", "MASK")
    assert MAIStitch.RETURN_NAMES == ("image", "width", "height", "mask")
    assert MAIStitch.INPUT_TYPES()["required"]["stitcher"][0] == MAIFastGPUResize.RETURN_TYPES[-1]
