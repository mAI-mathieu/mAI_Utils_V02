import pytest

from nodes.type_converter_node import MAITypeConverterNode


def convert(source_type="auto", boolean=False, string="", int=0, float=0.0,
            strict=False, **kwargs):
    node = MAITypeConverterNode()
    result = node.convert(source_type, boolean, string, int, float, strict, **kwargs)
    assert len(result) == 5
    assert result[4].seed == result[2] % (1 << 64)
    assert callable(result[4].generate_noise)
    return result[:4]


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


@pytest.mark.parametrize("value,expected", [
    (42, (True, "42", 42, 42.0)),
    (0, (False, "0", 0, 0.0)),
    (-7, (True, "-7", -7, -7.0)),
    (3.8, (True, "3.8", 3, 3.8)),
    (0.0, (False, "0.0", 0, 0.0)),
    (True, (True, "true", 1, 1.0)),
    (False, (False, "false", 0, 0.0)),
    ("12.7", (True, "12.7", 12, 12.7)),
    ("", (False, "", 0, 0.0)),
])
def test_connected_value_automatically_detects_source(value, expected):
    assert convert(value=value) == expected


def test_connected_value_overrides_saved_manual_source_and_other_fields():
    assert convert("string", string="different", int=99, float=9.9,
                   boolean=True, value=0) == (False, "0", 0, 0.0)


def test_disconnected_auto_source_uses_string_field():
    assert convert() == (False, "", 0, 0.0)
    assert convert(string="123") == (True, "123", 123, 123.0)


@pytest.mark.parametrize("value", [None, [], {}, object()])
def test_unsupported_connected_types_raise_clear_error(value):
    with pytest.raises(ValueError, match="value must be a STRING, INT, FLOAT or BOOLEAN"):
        convert(value=value)


@pytest.mark.parametrize("seed", [0, 1, (1 << 53) + 1, (1 << 64) - 1])
@pytest.mark.parametrize("as_text", [False, True])
def test_large_seed_precision_is_preserved(seed, as_text):
    result = MAITypeConverterNode().convert(value=str(seed) if as_text else seed)
    assert result[2] == seed
    assert result[4].seed == seed


@pytest.mark.parametrize("seed", [-1, -(1 << 64) - 5, 1 << 64, (1 << 64) + 9])
def test_noise_wraps_seed_without_changing_integer_output(seed):
    result = MAITypeConverterNode().convert(value=seed)
    assert result[2] == seed
    assert result[4].seed == seed % (1 << 64)


@pytest.mark.parametrize("value", ["12.7", "1e3", " 42 "])
def test_numeric_text_remains_supported_in_strict_mode(value):
    result = convert(value=value, strict=True)
    assert result[2] == int(float(value))


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "inf", "nan"])
def test_nonfinite_values_use_seed_zero_or_raise_clear_strict_error(value):
    assert convert(value=value)[2] == 0
    with pytest.raises(ValueError, match="Cannot convert .* to int"):
        convert(value=value, strict=True)


def test_auto_invalid_text_respects_strict_mode():
    assert convert(value="hello") == (False, "hello", 0, 0.0)
    with pytest.raises(ValueError, match="Cannot convert"):
        convert(value="hello", strict=True)


def test_unknown_manual_source_reports_error():
    with pytest.raises(ValueError, match="Unknown source_type"):
        convert("unknown")


def test_node_contract_preserves_existing_sockets_and_adds_noise_last():
    node = MAITypeConverterNode()
    assert node.CATEGORY == "mAI / Utils"
    assert node.FUNCTION == "convert"
    assert node.RETURN_TYPES == ("BOOLEAN", "STRING", "INT", "FLOAT", "NOISE")
    assert node.RETURN_NAMES == ("boolean", "string", "int", "float", "noise")
    inputs = node.INPUT_TYPES()
    assert tuple(inputs["required"]) == (
        "source_type", "boolean", "string", "int", "float", "strict",
    )
    assert inputs["required"]["source_type"][1]["default"] == "auto"
    assert inputs["required"]["int"][1]["max"] == (1 << 64) - 1
    assert inputs["optional"]["value"][0] == "*"
