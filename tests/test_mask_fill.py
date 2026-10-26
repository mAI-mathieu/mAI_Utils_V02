import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from nodes.mask_fill import MAIMaskFill
from utils.mask_fill import fill_masked_image


def test_hex_color_fills_only_the_mask_and_blends_soft_edges():
    image = torch.full((1, 2, 3, 3), 0.25, dtype=torch.float64)
    original = image.clone()
    mask = torch.tensor([[0, 0.5, 1], [1, 0, 0]], dtype=torch.float64)

    result = fill_masked_image(image, mask, color=" #fF8000 ")

    color = torch.tensor([1, 128 / 255, 0], dtype=image.dtype)
    torch.testing.assert_close(result[0, 0, 2], color)
    torch.testing.assert_close(result[0, 0, 1], (color + 0.25) / 2)
    assert torch.equal(result[0, 0, 0], image[0, 0, 0])
    assert torch.equal(image, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["color", "telea", "navier_stokes"])
def test_preserves_dtype_alpha_and_unmasked_precision_for_noncontiguous_image(dtype, method):
    if method != "color":
        pytest.importorskip("cv2")
    image = torch.rand(2, 5, 7, 4).to(dtype).transpose(1, 2)
    original = image.clone()
    mask = torch.zeros(1, 7, 5)
    mask[:, 2:5, 2:4] = 1

    result = fill_masked_image(image, mask, method, "#aBcDeF")

    assert result.shape == image.shape
    assert result.dtype == dtype
    assert result.device == image.device
    assert torch.equal(result[..., 3], image[..., 3])
    assert torch.equal(result[:, mask[0] == 0], image[:, mask[0] == 0])
    assert torch.equal(image, original)


def test_single_image_broadcasts_to_distinct_masks_and_clamps_mask_values():
    image = torch.full((1, 2, 2, 3), 0.25)
    mask = torch.tensor([[[-1, 2], [0, 0]], [[0, 0], [2, -1]]])
    result = fill_masked_image(image, mask, color="#FFFFFF")
    assert result.shape == (2, 2, 2, 3)
    assert torch.equal(result[0, 0, 0], image[0, 0, 0])
    assert torch.equal(result[0, 0, 1], torch.ones(3))
    assert torch.equal(result[1, 1, 0], torch.ones(3))
    assert torch.equal(result[1, 0, 1], image[0, 0, 1])


@pytest.mark.parametrize("method", ["telea", "navier_stokes"])
def test_inpainting_repairs_a_hole_without_changing_surroundings(method):
    pytest.importorskip("cv2")
    image = torch.empty(1, 11, 11, 3)
    background = torch.tensor([0.8, 0.4, 0.2])
    image[:] = background
    image[:, 4:7, 4:7] = 0
    mask = torch.zeros(1, 11, 11)
    mask[:, 4:7, 4:7] = 1

    # The irrelevant color value must not prevent inpainting.
    result = fill_masked_image(image, mask, method, color="unused")

    torch.testing.assert_close(result[0, 5, 5], background, atol=0.025, rtol=0)
    assert torch.equal(result[mask == 0], image[mask == 0])


@pytest.mark.parametrize("method", ["telea", "navier_stokes"])
def test_selected_algorithm_radius_and_soft_blend_match_opencv(method):
    cv2 = pytest.importorskip("cv2")
    image = torch.linspace(0, 1, 9 * 9 * 3).reshape(1, 9, 9, 3)
    image[:, 3:6, 3:6] = 0
    mask = torch.zeros(1, 9, 9)
    mask[:, 3:6, 3:6] = 0.5
    source = (image[0] * 255).round().to(torch.uint8).numpy()
    binary_mask = (mask[0] > 0).to(torch.uint8).numpy()
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    reference = torch.from_numpy(cv2.inpaint(source, binary_mask, 5.0, flag)).float() / 255

    result = fill_masked_image(image, mask, method, inpaint_radius=5.0)

    torch.testing.assert_close(result[0, 4, 4], (image[0, 4, 4] + reference[4, 4]) / 2)
    assert torch.equal(result[mask == 0], image[mask == 0])


@pytest.mark.parametrize("method", ["color", "telea", "navier_stokes"])
def test_empty_mask_is_a_copy_without_loading_opencv(monkeypatch, method):
    monkeypatch.setitem(sys.modules, "cv2", None)
    image = torch.rand(2, 4, 5, 3)
    result = fill_masked_image(image, torch.zeros(1, 4, 5), method)
    assert torch.equal(result, image)
    assert result.data_ptr() != image.data_ptr()


def test_color_fill_works_without_opencv_and_full_mask_is_allowed(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    image = torch.rand(1, 4, 5, 3)
    result = fill_masked_image(image, torch.ones(4, 5), color="#00FF00")
    assert torch.all(result[..., 0] == 0)
    assert torch.all(result[..., 1] == 1)
    assert torch.all(result[..., 2] == 0)


def test_missing_opencv_raises_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    mask = torch.zeros(1, 4, 5)
    mask[:, 1, 1] = 1
    with pytest.raises(RuntimeError, match="opencv-python-headless"):
        fill_masked_image(torch.rand(1, 4, 5, 3), mask, "telea")


@pytest.mark.parametrize("method", ["telea", "navier_stokes"])
def test_fully_masked_batch_item_cannot_be_inpainted(method):
    mask = torch.zeros(2, 4, 5)
    mask[1] = 1
    with pytest.raises(ValueError, match="unmasked source pixels"):
        fill_masked_image(torch.rand(2, 4, 5, 3), mask, method)


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), True, "3"])
def test_invalid_inpaint_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="inpaint_radius"):
        fill_masked_image(torch.zeros(1, 4, 5, 3), torch.zeros(4, 5), "telea", inpaint_radius=radius)


def test_invalid_method_color_shapes_batches_and_nonfinite_values_are_rejected():
    image = torch.zeros(2, 4, 5, 3)
    mask = torch.zeros(1, 4, 5)
    for kwargs, message in [({"method": "unknown"}, "Unknown"), ({"color": "red"}, "#RRGGBB")]:
        with pytest.raises(ValueError, match=message):
            fill_masked_image(image, mask, **kwargs)
    with pytest.raises(ValueError, match="dimensions must match"):
        fill_masked_image(image, torch.zeros(4, 6))
    with pytest.raises(ValueError, match="batch sizes must match"):
        fill_masked_image(image, torch.zeros(3, 4, 5))
    with pytest.raises(ValueError, match="RGB or RGBA"):
        fill_masked_image(torch.zeros(1, 4, 5, 1), mask)
    with pytest.raises(ValueError, match="finite"):
        fill_masked_image(image, mask.fill_(float("nan")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("method", ["color", "telea", "navier_stokes"])
def test_gpu_input_keeps_output_on_gpu_with_cpu_mask(method):
    if method != "color":
        pytest.importorskip("cv2")
    image = torch.rand(1, 7, 7, 3, device="cuda", dtype=torch.float16)
    mask = torch.zeros(7, 7)
    mask[2:5, 2:5] = 1
    result = fill_masked_image(image, mask, method)
    assert result.device == image.device
    assert result.dtype == image.dtype
    assert torch.equal(result[:, mask == 0], image[:, mask == 0])


def test_node_contract_and_whole_pack_registration(monkeypatch):
    inputs = MAIMaskFill.INPUT_TYPES()["required"]
    assert inputs["image"] == ("IMAGE",)
    assert inputs["mask"] == ("MASK",)
    assert inputs["method"][0] == ["color", "telea", "navier_stokes"]
    assert inputs["color"][1]["default"] == "#000000"
    assert MAIMaskFill.RETURN_TYPES == ("IMAGE",)
    assert MAIMaskFill.RETURN_NAMES == ("image",)
    assert MAIMaskFill.CATEGORY == "mAI / Image"
    output = MAIMaskFill().run(torch.zeros(1, 2, 2, 3), torch.ones(2, 2))
    assert isinstance(output, tuple) and len(output) == 1

    repo_dir = Path(__file__).resolve().parents[1]
    name = "test_mai_mask_fill_pack"
    spec = importlib.util.spec_from_file_location(
        name, repo_dir / "__init__.py", submodule_search_locations=[str(repo_dir)]
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIMaskFill"].__name__ == "MAIMaskFill"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIMaskFill"] == "mAI mask fill"
    assert "MAIMaskOutline" in package.NODE_CLASS_MAPPINGS
    assert "MAIImageBlur" in package.NODE_CLASS_MAPPINGS
