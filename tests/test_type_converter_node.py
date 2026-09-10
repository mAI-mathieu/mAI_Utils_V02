import pytest

from nodes.type_converter_node import MAITypeConverterNode


@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", ("true", 0, 0.0, True)),
        ("false", ("false", 0, 0.0, False)),
        ("12", ("12", 12, 12.0, True)),
        ("12.7", ("12.7", 12, 12.7, True)),
        ("", ("", 0, 0.0, False)),
        ("hello", ("hello", 0, 0.0, False)),
        (True, ("true", 1, 1.0, True)),
        (False, ("false", 0, 0.0, False)),
        (0, ("0", 0, 0.0, False)),
        (42, ("42", 42, 42.0, True)),
        (3.8, ("3.8", 3, 3.8, True)),
        (-3.8, ("-3.8", -3, -3.8, True)),
    ],
)
def test_autodetected_source_converts_to_single_selected_output(value, expected):
    node = MAITypeConverterNode()
    for output_type, result in zip(("string", "int", "float", "boolean"), expected):
        actual = node.convert(value, output_type)
        assert actual == (result,)
        assert type(actual[0]) is type(result)


@pytest.mark.parametrize("output_type", ["boolean", "int", "float"])
def test_invalid_string_raises_when_strict(output_type):
    with pytest.raises(ValueError, match=f"Cannot convert str to {output_type}"):
        MAITypeConverterNode().convert("hello", output_type, strict=True)


def test_strict_only_converts_selected_output():
    node = MAITypeConverterNode()
    assert node.convert("hello", "string", strict=True) == ("hello",)
    assert node.convert("true", "boolean", strict=True) == (True,)


def test_default_output_is_string():
    assert MAITypeConverterNode().convert(42) == ("42",)


def test_socket_contract():
    inputs = MAITypeConverterNode.INPUT_TYPES()["required"]
    assert set(inputs) == {"value", "output_type", "strict"}
    assert inputs["value"] == ("*", {"forceInput": True})
    assert MAITypeConverterNode.RETURN_TYPES == ("*",)
    assert MAITypeConverterNode.RETURN_NAMES == ("value",)
