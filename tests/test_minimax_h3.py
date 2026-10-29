import sys
from types import ModuleType

import pytest
import torch

from nodes.prepare_image_for_minimax_h3 import MAIPrepareImageForMinimaxH3
from utils.minimax_h3 import (
    MEGAPIXEL_OPTIONS,
    RESIZE_ALGORITHM_OPTIONS,
    RESIZE_MODE_MEGAPIXELS,
    RESIZE_MODE_OPTIONS,
    RESIZE_MODE_SHORT_SIDE,
    RESIZE_MODE_SHORT_SIDE_768,
    calculate_minimax_h3_dimensions,
    calculate_short_side_dimensions,
    parse_target_megapixels,
)


def test_megapixel_options_cover_requested_range():
    assert MEGAPIXEL_OPTIONS[0] == "0.2 MP"
    assert MEGAPIXEL_OPTIONS[-1] == "8.0 MP"
    assert len(MEGAPIXEL_OPTIONS) == 79


@pytest.mark.parametrize(
    ("width", "height", "megapixels", "expected"),
    [
        (1920, 1080, "1.0 MP", (1376, 768)),
        (1080, 1920, "1.0 MP", (768, 1376)),
        (1024, 1024, "0.2 MP", (448, 448)),
        (1024, 1024, "2.0 MP", (1440, 1440)),
        (2048, 1024, "8.0 MP", (4096, 2048)),
        (1024, 2048, "8.0 MP", (2048, 4096)),
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


@pytest.mark.parametrize(
    ("width", "height", "short_side", "multiple_of", "expected"),
    [
        (1920, 1080, 777, 1, (1381, 777)),
        (1080, 1920, 777, 1, (777, 1381)),
        (1920, 1080, 777, 32, (1376, 768)),
        (1920, 1080, 777, 16, (1376, 784)),
        (1920, 1080, 777, 7, (1379, 777)),
        (1920, 1080, 1, 32, (32, 32)),
        (1024, 1024, 777, 1, (777, 777)),
    ],
)
def test_short_side_accepts_arbitrary_positive_integers(
    width, height, short_side, multiple_of, expected
):
    assert calculate_short_side_dimensions(width, height, short_side, multiple_of) == expected


@pytest.mark.parametrize("multiple_of", [1, 7, 16, 32, 64])
def test_megapixel_mode_uses_selected_multiple(multiple_of):
    width, height = calculate_minimax_h3_dimensions(1920, 1080, "1.0 MP", multiple_of)

    assert width >= multiple_of
    assert height >= multiple_of
    assert width % multiple_of == 0
    assert height % multiple_of == 0


@pytest.mark.parametrize("value", [0, -1, 1.5, "32", None, True])
def test_rejects_invalid_grid_multiple_in_both_modes(value):
    with pytest.raises(ValueError, match="multiple_of must be a positive integer"):
        calculate_minimax_h3_dimensions(1920, 1080, "1.0 MP", multiple_of=value)
    with pytest.raises(ValueError, match="multiple_of must be a positive integer"):
        calculate_short_side_dimensions(1920, 1080, multiple_of=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, "777", None, True])
def test_rejects_invalid_short_side_target(value):
    with pytest.raises(ValueError, match="short_side must be a positive integer"):
        calculate_short_side_dimensions(1920, 1080, short_side=value)


def test_parses_dropdown_label_and_numeric_value():
    assert parse_target_megapixels("0.7 MP") == 0.7
    assert parse_target_megapixels(0.7) == 0.7


@pytest.mark.parametrize("value", [0.2, 0.25, 0.725, 1.25, 2.0, 2.001, 4.5, 7.99, 8.0])
def test_accepts_float_targets_between_old_dropdown_steps(value):
    assert parse_target_megapixels(value) == value
    width, height = calculate_minimax_h3_dimensions(1920, 1080, value, multiple_of=1)
    assert width * height == pytest.approx(value * 1024 * 1024, rel=0.005)


def test_node_exposes_float_megapixel_target():
    input_type, options = MAIPrepareImageForMinimaxH3.INPUT_TYPES()["required"]["target_megapixels"]
    assert input_type == "FLOAT"
    assert options == {"default": 1.0, "min": 0.2, "max": 8.0, "step": 0.01}
    image = torch.zeros(1, 512, 512, 3)
    output, = MAIPrepareImageForMinimaxH3().prepare(image, 0.25)
    assert output is image


@pytest.mark.parametrize(
    ("kwargs", "expected_width"),
    [({}, 1376), ({"multiple_of": 64}, 1344), ({"multiple_of": 1}, 1365)],
)
@pytest.mark.parametrize("algorithm", [None, *RESIZE_ALGORITHM_OPTIONS])
def test_node_resizes_with_selected_comfyui_algorithm(
    monkeypatch, kwargs, expected_width, algorithm
):
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
    options = dict(kwargs)
    if algorithm is not None:
        options["resize_algorithm"] = algorithm
    output, = MAIPrepareImageForMinimaxH3().prepare(image, "1.0 MP", **options)

    assert output.shape == (2, 768, expected_width, 3)
    assert calls == {
        "width": expected_width,
        "height": 768,
        "method": algorithm or "lanczos",
        "crop": "disabled",
    }


@pytest.mark.parametrize(
    ("kwargs", "expected_width", "expected_height"),
    [
        ({}, 1280, 768),
        ({"short_side": 777, "multiple_of": 1}, 1305, 777),
        ({"short_side": 777, "multiple_of": 16}, 1312, 784),
        ({"short_side": 777, "multiple_of": 1, "resize_algorithm": "bicubic"}, 1305, 777),
    ],
)
@pytest.mark.parametrize("resize_mode", [RESIZE_MODE_SHORT_SIDE, RESIZE_MODE_SHORT_SIDE_768])
def test_node_short_side_mode_bypasses_invalid_megapixel_value(
    monkeypatch, kwargs, expected_width, expected_height, resize_mode
):
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
        resize_mode,
        **kwargs,
    )

    assert output.shape == (1, expected_height, expected_width, 3)
    assert calls == {
        "width": expected_width,
        "height": expected_height,
        "method": kwargs.get("resize_algorithm", "lanczos"),
        "crop": "disabled",
    }


def test_node_keeps_megapixel_mode_as_default():
    inputs = MAIPrepareImageForMinimaxH3.INPUT_TYPES()["required"]

    assert inputs["resize_mode"][1]["default"] == RESIZE_MODE_MEGAPIXELS


def test_node_shows_short_side_label_and_accepts_legacy_api_value():
    inputs = MAIPrepareImageForMinimaxH3.INPUT_TYPES()

    assert inputs["required"]["resize_mode"][0] == ["Target megapixels", "Short side"]
    for mode in (*RESIZE_MODE_OPTIONS, RESIZE_MODE_SHORT_SIDE_768):
        assert MAIPrepareImageForMinimaxH3.VALIDATE_INPUTS(mode) is True
    assert MAIPrepareImageForMinimaxH3.VALIDATE_INPUTS("invalid") == "Unknown resize mode: invalid"
    algorithms, options = inputs["optional"]["resize_algorithm"]
    assert algorithms == ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]
    assert options["default"] == "lanczos"


@pytest.mark.parametrize("algorithm", ["invalid", "", None])
def test_node_rejects_unknown_resize_algorithm_even_when_size_is_unchanged(algorithm):
    image = torch.rand(1, 768, 768, 3)

    with pytest.raises(ValueError, match="Unknown resize algorithm"):
        MAIPrepareImageForMinimaxH3().prepare(
            image, "1.0 MP", RESIZE_MODE_SHORT_SIDE, resize_algorithm=algorithm
        )


def test_node_exposes_integer_targets_with_compatible_defaults():
    inputs = MAIPrepareImageForMinimaxH3.INPUT_TYPES()

    assert list(inputs["required"]) == ["image", "target_megapixels", "resize_mode"]
    for name, default in (("short_side", 768), ("multiple_of", 32)):
        input_type, options = inputs["optional"][name]
        assert input_type == "INT"
        assert options["default"] == default
        assert options["min"] == 1
        assert options["step"] == 1
    assert MAIPrepareImageForMinimaxH3.RETURN_TYPES == ("IMAGE",)
    assert MAIPrepareImageForMinimaxH3.RETURN_NAMES == ("image",)


def test_node_rejects_unknown_resize_mode():
    image = torch.rand(1, 32, 32, 3)

    with pytest.raises(ValueError, match="Unknown resize mode"):
        MAIPrepareImageForMinimaxH3().prepare(image, "1.0 MP", "invalid")


@pytest.mark.parametrize(
    "value",
    ["0.1 MP", "8.1 MP", "invalid", None, 0.199, 8.001, float("nan"), float("inf"), True, False],
)
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
