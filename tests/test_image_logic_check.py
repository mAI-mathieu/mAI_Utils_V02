from types import SimpleNamespace

import pytest

from nodes.image_logic_check import MAIImageLogicCheck
from utils.image_logic_check import compare_values, image_property_value


@pytest.mark.parametrize("property,expected", [
    ("Megapixels", 2.0736),
    ("Width", 1920.0),
    ("Height", 1080.0),
    ("Aspect Ratio", 1920 / 1080),
    ("Batch Size", 4.0),
])
def test_properties_use_bhwc_batch_dimensions(property, expected):
    result = image_property_value((4, 1080, 1920, 3), property)
    assert result == pytest.approx(expected)
    assert type(result) is float


@pytest.mark.parametrize("operator,expected", [
    (">", False), (">=", False), ("<", True), ("<=", True),
    ("==", False), ("!=", True),
])
def test_comparisons_with_unequal_values(operator, expected):
    assert compare_values(1.048576, operator, 2.2) is expected


@pytest.mark.parametrize("operator,expected", [
    (">", False), (">=", True), ("<", False), ("<=", True),
    ("==", True), ("!=", False),
])
def test_comparisons_at_boundary(operator, expected):
    assert compare_values(2.2, operator, 2.2) is expected


@pytest.mark.parametrize("actual,threshold,close", [
    (0.0, 0.0000005, True),  # Absolute tolerance near zero.
    (1024.0, 1024.0005, True),  # Relative tolerance for larger values.
    (1.0, 1.00001, False),
])
def test_equality_tolerances_and_inequality_inverse(actual, threshold, close):
    assert compare_values(actual, "==", threshold) is close
    assert compare_values(actual, "!=", threshold) is (not close)


def test_ordered_comparisons_do_not_use_equality_tolerance():
    assert compare_values(1.0, "<", 1.0000005) is True
    assert compare_values(1.0000005, ">", 1.0) is True


@pytest.mark.parametrize("shape", [
    None, (), (1080, 1920, 3), (1, 1, 1080, 1920, 3),
    (0, 1080, 1920, 3), (1, 0, 1920, 3), (1, -1, 1920, 3),
    (1, 1080, 0, 3), (1, 1080, 1920, 0),
])
def test_rejects_missing_or_empty_shape(shape):
    with pytest.raises(ValueError, match="non-empty shape"):
        MAIImageLogicCheck().run(SimpleNamespace(shape=shape), "Aspect Ratio", ">", 2.2)


def test_rejects_missing_image():
    with pytest.raises(ValueError, match="non-empty shape"):
        MAIImageLogicCheck().run(None, "Width", ">", 2.2)


def test_rejects_unknown_property_and_operator():
    with pytest.raises(ValueError, match="Unknown image property: Depth"):
        image_property_value((1, 100, 100, 3), "Depth")
    with pytest.raises(ValueError, match="Unknown comparison operator: contains"):
        compare_values(1.0, "contains", 2.2)


def test_node_reads_only_shape_and_returns_native_types():
    class ShapeOnlyImage:
        def __getattribute__(self, name):
            if name == "shape":
                return (4, 1024, 1024, 4)
            raise AssertionError(f"Unexpected image access: {name}")

        def __setattr__(self, name, value):
            raise AssertionError(f"Unexpected image mutation: {name}")

    result, actual_value = MAIImageLogicCheck().run(
        ShapeOnlyImage(), "Megapixels", ">", 2.2
    )
    assert result is False
    assert actual_value == 1.048576
    assert type(result) is bool
    assert type(actual_value) is float


def test_node_contract_and_defaults():
    node = MAIImageLogicCheck()
    inputs = node.INPUT_TYPES()["required"]
    assert list(inputs) == ["image", "property", "operator", "compare_value"]
    assert inputs["image"] == ("IMAGE",)
    assert inputs["property"][0] == [
        "Megapixels", "Width", "Height", "Aspect Ratio", "Batch Size",
    ]
    assert inputs["operator"][0] == [">", ">=", "<", "<=", "==", "!="]
    assert inputs["compare_value"] == (
        "FLOAT", {"default": 2.2, "min": 0.0, "max": 1e15, "step": 0.01},
    )
    assert node.RETURN_TYPES == ("BOOLEAN", "FLOAT")
    assert node.RETURN_NAMES == ("result", "actual_value")
    assert node.CATEGORY == "mAI / Logic"
    defaults = {name: spec[1]["default"] for name, spec in inputs.items() if name != "image"}
    assert getattr(node, node.FUNCTION)(SimpleNamespace(shape=(1, 1024, 1024, 3)), **defaults) == (
        False, 1.048576,
    )
