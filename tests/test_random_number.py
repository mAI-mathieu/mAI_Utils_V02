import math
import sys
from types import ModuleType

import pytest

from nodes.random_number import MAIRandomNumber


@pytest.mark.parametrize("use_upper_bound", [False, True])
def test_both_endpoints_can_be_drawn(use_upper_bound, monkeypatch):
    monkeypatch.setattr(
        "nodes.random_number.random.randrange", lambda stop: stop - 1 if use_upper_bound else 0
    )
    monkeypatch.setattr(
        "nodes.random_number.random.randint", lambda low, high: high if use_upper_bound else low
    )

    integer_value, float_value = MAIRandomNumber().run(-4, 3)

    expected = 3 if use_upper_bound else -4
    assert integer_value == expected and type(integer_value) is int
    assert float_value == expected and type(float_value) is float


def test_fractional_bounds_keep_integer_inside_range(monkeypatch):
    def draw_integer(low, high):
        assert (low, high) == (-4, -3)
        return high

    monkeypatch.setattr("nodes.random_number.random.randint", draw_integer)
    monkeypatch.setattr("nodes.random_number.random.randrange", lambda stop: stop // 2)

    assert MAIRandomNumber().run(-4.5, -2.75) == (-3, -3.625)


def test_range_without_whole_numbers_still_draws_a_float(monkeypatch):
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

    graph_utils = ModuleType("comfy_execution.graph_utils")
    graph_utils.ExecutionBlocker = ExecutionBlocker
    monkeypatch.setitem(sys.modules, graph_utils.__name__, graph_utils)

    integer_value, float_value = MAIRandomNumber().run(0.1, 0.9)

    assert isinstance(integer_value, ExecutionBlocker)
    assert "no whole number" in integer_value.message
    assert 0.1 <= float_value <= 0.9


def test_equal_integer_bounds_return_the_bound():
    assert MAIRandomNumber().run(5, 5) == (5, 5.0)


def test_wide_finite_range_can_be_sampled(monkeypatch):
    monkeypatch.setattr("nodes.random_number.random.randrange", lambda stop: stop // 2)
    minimum, maximum = -sys.float_info.max, sys.float_info.max

    integer_value, float_value = MAIRandomNumber().run(minimum, maximum)

    assert minimum <= integer_value <= maximum
    assert float_value == 0.0


@pytest.mark.parametrize("minimum,maximum", [(math.nan, 1), (0, math.inf), (-math.inf, 0)])
def test_non_finite_bounds_are_rejected(minimum, maximum):
    with pytest.raises(ValueError, match="must be finite"):
        MAIRandomNumber().run(minimum, maximum)


def test_reversed_bounds_are_rejected():
    with pytest.raises(ValueError, match="minimum must be less than or equal to maximum"):
        MAIRandomNumber().run(2, 1)


def test_each_queue_refreshes_random_values():
    assert math.isnan(MAIRandomNumber.IS_CHANGED(minimum=0.0, maximum=1.0))
