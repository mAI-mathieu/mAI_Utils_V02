import pytest

from nodes.int_to_nearest_multiple import MAIIntToNearestMultiple
from utils.integer_multiples import round_to_multiple


@pytest.mark.parametrize(
    "value,multiple,down,up",
    [
        (100, 64, 64, 128),
        (128, 64, 128, 128),
        (0, 8, 0, 0),
        (3, 8, 0, 8),
        (-3, 8, -8, 0),
        (-100, 64, -128, -64),
        (-128, 64, -128, -128),
        (42, 1, 42, 42),
        (2**60 + 1, 8, 2**60, 2**60 + 8),
        (-(2**60 + 1), 8, -(2**60 + 8), -(2**60)),
    ],
)
def test_directional_rounding(value, multiple, down, up):
    assert round_to_multiple(value, multiple, "round down") == down
    assert round_to_multiple(value, multiple, "round up") == up


@pytest.mark.parametrize("multiple", [0, -1, -8, 1.5, "8", None, True])
def test_rejects_invalid_divisor(multiple):
    with pytest.raises(ValueError, match="multiple must be a positive integer"):
        round_to_multiple(100, multiple)


@pytest.mark.parametrize("value", [1.5, "100", None, True])
def test_rejects_noninteger_value(value):
    with pytest.raises(ValueError, match="value must be an integer"):
        round_to_multiple(value, 8)


def test_rejects_unknown_rounding_mode():
    with pytest.raises(ValueError, match="Unknown rounding mode"):
        round_to_multiple(100, 8, "nearest")


def test_results_are_divisible_and_bracket_input():
    for value in range(-100, 101):
        for multiple in range(1, 20):
            down = round_to_multiple(value, multiple, "round down")
            up = round_to_multiple(value, multiple, "round up")
            assert down % multiple == up % multiple == 0
            assert down <= value <= up
            assert value - down < multiple
            assert up - value < multiple


def test_node_execution_and_defaults():
    node = MAIIntToNearestMultiple()
    inputs = node.INPUT_TYPES()["required"]
    defaults = {name: spec[1]["default"] for name, spec in inputs.items()}
    assert node.run(**defaults) == (0,)
    assert node.run(100, 64) == (64,)
    assert node.run(100, 64, "round up") == (128,)
    assert node.RETURN_TYPES == ("INT",)
    assert node.RETURN_NAMES == ("value",)
    assert node.CATEGORY == "mAI / Utils"


def test_node_validates_connected_divisor():
    with pytest.raises(ValueError, match="multiple must be a positive integer"):
        MAIIntToNearestMultiple().run(100, 0)
