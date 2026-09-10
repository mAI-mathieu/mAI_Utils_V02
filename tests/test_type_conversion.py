import pytest

from utils.type_conversion import convert_value


@pytest.mark.parametrize("value", [" YES ", "on", "y", "-2", "0.5"])
def test_truthy_strings(value):
    assert convert_value(value, "boolean", strict=True) is True


@pytest.mark.parametrize("value", [" NO ", "off", "n", "0.0", "  "])
def test_false_strings(value):
    assert convert_value(value, "boolean", strict=True) is False


def test_integer_string_preserves_seed_precision():
    assert convert_value("18446744073709551615", "int") == 18446744073709551615


@pytest.mark.parametrize("value", ["inf", "nan", float("inf"), float("nan")])
def test_invalid_integer_has_fallback_or_clear_error(value):
    assert convert_value(value, "int") == 0
    with pytest.raises(ValueError, match="Cannot convert"):
        convert_value(value, "int", strict=True)


@pytest.mark.parametrize("value", [None, [], {}, object()])
def test_unsupported_sources_raise_clear_errors(value):
    with pytest.raises(ValueError, match="input must be a string, int, float, or boolean"):
        convert_value(value)


def test_unknown_output_type_raises():
    with pytest.raises(ValueError, match="Unknown output_type"):
        convert_value("hello", "image")
