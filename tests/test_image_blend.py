import pytest
import torch

from nodes.image_blend import MAIImageBlend
from utils.image_blend import BLEND_MODES, blend_image_batches, parse_blend_percentages


def _images(*values, dtype=torch.float32, device="cpu"):
    return torch.tensor(values, dtype=dtype, device=device).reshape(-1, 1, 1, 1).expand(-1, 2, 3, 3)


def test_parse_blend_percentages_preserves_order_and_accepts_decimals():
    assert parse_blend_percentages("100%, 25.5\n0 %\n25.5%") == [100, 25.5, 0, 25.5]


@pytest.mark.parametrize("value", ["", " , \n", "red", "50%%", "nan", "inf", "-1", "101"])
def test_invalid_blend_percentages_are_rejected(value):
    with pytest.raises(ValueError, match="percentage"):
        parse_blend_percentages(value)


def test_legacy_list_still_produces_one_image_per_percentage():
    top = _images(0.8)
    bottom = _images(0.2)
    result, = MAIImageBlend().run(top, bottom, blend_list="0%, 50%, 100%")
    torch.testing.assert_close(result, _images(0.2, 0.5, 0.8))
    assert torch.equal(result[:1], bottom)
    assert torch.equal(result[-1:], top)


@pytest.mark.parametrize(("top", "bottom", "expected"), [
    ((1,), (0, 0.5), (0.375, 0.6875)),
    ((0.5, 1), (0,), (0.1875, 0.375)),
    ((0, 1), (1, 0.5), (0.625, 0.6875)),
])
def test_one_fractional_percentage_applies_to_the_batch(top, bottom, expected):
    result = blend_image_batches(_images(*top), _images(*bottom), 37.5)
    torch.testing.assert_close(result, _images(*expected))


@pytest.mark.parametrize(("mode", "expected"), [
    ("normal", (0.7, 0.3)),
    ("multiply", (0.14, 0.18)),
    ("screen", (0.76, 0.72)),
    ("overlay", (0.28, 0.44)),
    ("soft_light", (0.2992, 0.504)),
    ("difference", (0.5, 0.3)),
])
def test_blend_modes_with_zero_partial_and_full_strength(mode, expected):
    bottom = _images(0.2, 0.6)
    top = _images(0.7, 0.3)
    full = _images(*expected)
    assert torch.equal(blend_image_batches(top, bottom, 0.0, mode), bottom)
    torch.testing.assert_close(blend_image_batches(top, bottom, 100.0, mode), full)
    torch.testing.assert_close(blend_image_batches(top, bottom, 25.0, mode), 0.75 * bottom + 0.25 * full)


@pytest.mark.parametrize("value", [-1, 100.1, float("nan"), float("inf"), "50%", True])
def test_invalid_float_percentage_fails_clearly(value):
    with pytest.raises(ValueError, match="blend_percentage.*0 to 100"):
        blend_image_batches(_images(1), _images(0), value)


def test_unknown_mode_fails_clearly():
    with pytest.raises(ValueError, match="Unknown blend mode"):
        blend_image_batches(_images(1), _images(0), 50.0, "unknown")


def test_node_contract_and_default():
    inputs = MAIImageBlend.INPUT_TYPES()
    assert list(inputs["required"]) == ["image_top", "image_bottom"]
    assert tuple(data[0] for data in inputs["required"].values()) == ("IMAGE", "IMAGE")
    options, mode_settings = inputs["optional"]["blend_mode"]
    assert tuple(options) == BLEND_MODES
    assert mode_settings["default"] == "normal"
    input_type, percentage_settings = inputs["optional"]["blend_percentage"]
    assert input_type == "FLOAT"
    assert [percentage_settings[key] for key in ("min", "max", "default")] == [0.0, 100.0, 50.0]
    assert inputs["optional"]["blend_list"][1]["hidden"] is True
    assert MAIImageBlend.RETURN_TYPES == ("IMAGE",)
    assert MAIImageBlend.RETURN_NAMES == ("images",)
    result, = MAIImageBlend().run(_images(1), _images(0))
    torch.testing.assert_close(result, _images(0.5))


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_blend_preserves_bottom_device_and_dtype(device):
    result = blend_image_batches(_images(0.5), _images(1, dtype=torch.float16, device=device), 50.0)
    torch.testing.assert_close(result, _images(0.75, dtype=torch.float16, device=device))


@pytest.mark.parametrize(("top_count", "bottom_count", "percent_count"), [(2, 3, 1), (2, 2, 3)])
def test_mismatched_counts_fail_clearly(top_count, bottom_count, percent_count):
    with pytest.raises(ValueError, match="same number of items"):
        blend_image_batches(_images(*([0] * top_count)), _images(*([1] * bottom_count)),
                            blend_list=",".join(["50"] * percent_count))


@pytest.mark.parametrize(("top", "message"), [
    (None, "image_top.*IMAGE batch"),
    (torch.empty(0, 2, 3, 3), "image_top.*IMAGE batch"),
    (torch.zeros(1, 2, 2, 3), "matching height, width, and channels"),
    (torch.zeros(1, 2, 3, 4), "matching height, width, and channels"),
])
def test_invalid_images_fail_clearly(top, message):
    with pytest.raises(ValueError, match=message):
        blend_image_batches(top, _images(0))
