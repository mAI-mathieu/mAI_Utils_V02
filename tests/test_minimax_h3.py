import sys
from types import ModuleType

import pytest
import torch

from nodes.prepare_image_for_minimax_h3 import MAIPrepareImageForMinimaxH3
from utils.minimax_h3 import (
    MEGAPIXEL_OPTIONS,
    RESIZE_MODE_MEGAPIXELS,
    RESIZE_MODE_SHORT_SIDE_768,
    calculate_minimax_h3_dimensions,
    calculate_short_side_dimensions,
    parse_target_megapixels,
)


def test_megapixel_options_cover_requested_range():
    assert MEGAPIXEL_OPTIONS[0] == "0.2 MP"
    assert MEGAPIXEL_OPTIONS[-1] == "2.0 MP"
    assert len(MEGAPIXEL_OPTIONS) == 19


@pytest.mark.parametrize(
    ("width", "height", "megapixels", "expected"),
    [
        (1920, 1080, "1.0 MP", (1376, 768)),
        (1080, 1920, "1.0 MP", (768, 1376)),
        (1024, 1024, "0.2 MP", (448, 448)),
        (1024, 1024, "2.0 MP", (1440, 1440)),
    ],
)
def test_calculates_expected_grid_dimensions(width, height, megapixels, expected):
    assert calculate_minimax_h3_dimensions(width, height, megapixels) == expected


@pytest.mark.parametrize("megapixels", MEGAPIXEL_OPTIONS)
def test_all_options_produce_multiples_of_32(megapixels):
    width, height = calculate_minimax_h3_dimensions(1379, 821, megapixels)

    assert width % 32 == 0
    assert height % 32 == 0


def test_output_area_is_close_to_selected_megapixels():
    width, height = calculate_minimax_h3_dimensions(1920, 1080, "1.0 MP")

    assert width * height == pytest.approx(1024 * 1024, rel=0.05)


def test_aspect_ratio_stays_close_to_source():
    source_ratio = 1920 / 1080
    width, height = calculate_minimax_h3_dimensions(1920, 1080, "1.0 MP")

    assert width / height == pytest.approx(source_ratio, rel=0.03)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1920, 1080, (1376, 768)),
        (1080, 1920, (768, 1376)),
        (1024, 1024, (768, 768)),
        (1379, 821, (1280, 768)),
    ],
)
def test_short_side_mode_calculates_expected_grid_dimensions(width, height, expected):
    assert calculate_short_side_dimensions(width, height) == expected


def test_short_side_mode_is_768_and_both_dimensions_are_multiples_of_32():
    width, height = calculate_short_side_dimensions(1379, 821)

    assert min(width, height) == 768
    assert width % 32 == 0
    assert height % 32 == 0


def test_parses_dropdown_label_and_numeric_value():
    assert parse_target_megapixels("0.7 MP") == 0.7
    assert parse_target_megapixels(0.7) == 0.7


def test_node_resizes_with_comfyui_lanczos(monkeypatch):
    calls = {}
    comfy_module = ModuleType("comfy")
    comfy_utils_module = ModuleType("comfy.utils")

    def common_upscale(samples, width, height, method, crop):
        calls.update(width=width, height=height, method=method, crop=crop)
        return torch.zeros(
            samples.shape[0],
            samples.shape[1],
            height,
            width,
            dtype=samples.dtype,
        )

    comfy_utils_module.common_upscale = common_upscale
    comfy_module.utils = comfy_utils_module
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)

    image = torch.rand(2, 108, 192, 3)
    output, = MAIPrepareImageForMinimaxH3().prepare(image, "1.0 MP")

    assert output.shape == (2, 768, 1376, 3)
    assert calls == {
        "width": 1376,
        "height": 768,
        "method": "lanczos",
        "crop": "disabled",
    }


def test_node_short_side_mode_bypasses_invalid_megapixel_value(monkeypatch):
    calls = {}
    comfy_module = ModuleType("comfy")
    comfy_utils_module = ModuleType("comfy.utils")

    def common_upscale(samples, width, height, method, crop):
        calls.update(width=width, height=height, method=method, crop=crop)
        return torch.zeros(
            samples.shape[0],
            samples.shape[1],
            height,
            width,
            dtype=samples.dtype,
        )

    comfy_utils_module.common_upscale = common_upscale
    comfy_module.utils = comfy_utils_module
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)

    image = torch.rand(1, 821, 1379, 3)
    output, = MAIPrepareImageForMinimaxH3().prepare(
        image,
        "not used",
        RESIZE_MODE_SHORT_SIDE_768,
    )

    assert output.shape == (1, 768, 1280, 3)
    assert calls == {
        "width": 1280,
        "height": 768,
        "method": "lanczos",
        "crop": "disabled",
    }


def test_node_keeps_megapixel_mode_as_default():
    inputs = MAIPrepareImageForMinimaxH3.INPUT_TYPES()["required"]

    assert inputs["resize_mode"][1]["default"] == RESIZE_MODE_MEGAPIXELS


def test_node_rejects_unknown_resize_mode():
    image = torch.rand(1, 32, 32, 3)

    with pytest.raises(ValueError, match="Unknown resize mode"):
        MAIPrepareImageForMinimaxH3().prepare(image, "1.0 MP", "invalid")


@pytest.mark.parametrize("value", ["0.1 MP", "2.1 MP", "invalid", None])
def test_rejects_invalid_megapixel_values(value):
    with pytest.raises(ValueError):
        parse_target_megapixels(value)


@pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-1, 100)])
def test_rejects_invalid_source_dimensions(width, height):
    with pytest.raises(ValueError):
        calculate_minimax_h3_dimensions(width, height, "1.0 MP")


@pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-1, 100)])
def test_short_side_mode_rejects_invalid_source_dimensions(width, height):
    with pytest.raises(ValueError):
        calculate_short_side_dimensions(width, height)
