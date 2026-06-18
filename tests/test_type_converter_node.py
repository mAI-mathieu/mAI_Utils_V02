import pytest

from nodes.type_converter_node import MAITypeConverterNode


def convert(source_type, boolean=False, string="", int=0, float=0.0, strict=False):
    node = MAITypeConverterNode()
    return node.convert(source_type, boolean, string, int, float, strict)


def test_string_true_converts_to_all_outputs():
    result = convert("string", string="true")
    assert result == (True, "true", 0, 0.0)


def test_string_false_converts_to_all_outputs():
    result = convert("string", string="false")
    assert result == (False, "false", 0, 0.0)


def test_string_integer_converts_to_all_outputs():
    result = convert("string", string="12")
    assert result == (True, "12", 12, 12.0)


def test_string_float_converts_to_all_outputs():
    result = convert("string", string="12.7")
    assert result == (True, "12.7", 12, 12.7)


def test_empty_string_converts_to_false_defaults():
    result = convert("string", string="")
    assert result == (False, "", 0, 0.0)


def test_invalid_string_returns_defaults_when_not_strict():
    result = convert("string", string="hello", strict=False)
    assert result == (False, "hello", 0, 0.0)


def test_invalid_string_raises_when_strict():
    with pytest.raises(ValueError):
        convert("string", string="hello", strict=True)


def test_boolean_true_converts_to_all_outputs():
    result = convert("boolean", boolean=True)
    assert result == (True, "true", 1, 1.0)


def test_int_zero_converts_to_all_outputs():
    result = convert("int", int=0)
    assert result == (False, "0", 0, 0.0)


def test_int_nonzero_converts_to_all_outputs():
    result = convert("int", int=42)
    assert result == (True, "42", 42, 42.0)


def test_float_converts_to_all_outputs():
    result = convert("float", float=3.8)
    assert result == (True, "3.8", 3, 3.8)
