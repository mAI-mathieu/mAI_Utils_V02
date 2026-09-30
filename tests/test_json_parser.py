import json

import pytest

from nodes.json_parser import MAIJsonParser
from utils.json_parser import parse_json_path, select_json_value


@pytest.mark.parametrize("value", ["a forest", "123", 1024, -7, 1.25, True, False, "", 0])
def test_auto_preserves_scalar_value_and_type(value):
    result = select_json_value(json.dumps({"value": value}), "value")
    assert result == value
    assert type(result) is type(value)


@pytest.mark.parametrize("path", ["items[0].settings.width", "$.items[0].settings.width", '["items"][0]["settings"].width'])
def test_nested_objects_and_arrays(path):
    text = '{"items":[{"settings":{"width":1024}}]}'
    assert select_json_value(text, path) == 1024


@pytest.mark.parametrize("key", ["a.b", "a[b]", "0", "", " padded ", 'a"b', "a\\b", "$", "日本語"])
def test_literal_keys(key):
    assert select_json_value(json.dumps({key: "found"}), f"[{json.dumps(key)}]") == "found"


@pytest.mark.parametrize("path", ["", "  ", "$"])
@pytest.mark.parametrize("value", [{"text": "café"}, [1, True], None])
def test_root_containers_and_null_are_json_text(path, value):
    result = select_json_value(json.dumps(value), path)
    assert isinstance(result, str)
    assert json.loads(result) == value


def test_root_array_and_whitespace_in_brackets():
    assert select_json_value('[{"name":"first"}]', '[ 0 ][ "name" ]') == "first"


@pytest.mark.parametrize("value,expected", [(True, "true"), (False, "false"), (123, "123"), (1.5, "1.5"), (None, "null"), ("hello", "hello")])
def test_string_mode(value, expected):
    assert select_json_value(json.dumps(value), "", "string") == expected


@pytest.mark.parametrize("text", ["", "  ", '{"a":}', "```json\n{}\n```", "{", "NaN", "Infinity", "-Infinity", "1e999", None])
def test_invalid_json_is_rejected(text):
    with pytest.raises(ValueError):
        select_json_value(text)


def test_invalid_json_reports_location():
    with pytest.raises(ValueError, match="line 2, column"):
        select_json_value('{\n"a": }')


@pytest.mark.parametrize("path", [".a", "a.", "a..b", "a.[0]", "a[", "a[]", "a[-1]", "a[1.5]", "a[*]", "a['b']", 'a["b]', 'a["b"', "a[0]b", "$a", "a]", "a. b"])
def test_malformed_paths_are_rejected(path):
    with pytest.raises(ValueError):
        parse_json_path(path)


@pytest.mark.parametrize("text,path,message", [
    ('{"a":1}', "missing", "was not found"),
    ('{"a":1}', "a.b", "requires an object"),
    ('{"a":1}', "a[0]", "requires an array"),
    ('{"a":[]}', "a[0]", "out of range"),
    ('{"a":[1]}', "a[1]", "out of range"),
    ('{"a":null}', "a.b", "requires an object"),
    ('{"0":"key"}', "[0]", "requires an array"),
    ('["item"]', '["0"]', "requires an object"),
])
def test_missing_or_incompatible_path_is_an_error(text, path, message):
    with pytest.raises(ValueError, match=message):
        select_json_value(text, path)


def test_unknown_output_mode_and_nontext_path_are_rejected():
    with pytest.raises(ValueError, match="Unknown JSON output type"):
        select_json_value("{}", "", "unsupported")
    with pytest.raises(ValueError, match="must be text"):
        select_json_value("{}", None)


def test_duplicate_keys_follow_standard_json_last_value_behavior():
    assert select_json_value('{"a":1,"a":2}', "a") == 2


def test_node_contract_defaults_and_connected_inputs():
    node = MAIJsonParser()
    inputs = node.INPUT_TYPES()["required"]
    assert node.RETURN_TYPES == ("*",)
    assert node.RETURN_NAMES == ("value",)
    assert node.CATEGORY == "mAI / Text"
    defaults = {name: spec[1]["default"] for name, spec in inputs.items()}
    assert node.run(**defaults) == ("a cinematic landscape",)
    assert node.run('{"width":1024}', "width") == (1024,)
    assert node.run('{"width":1024}', "width", "string") == ("1024",)
    assert node.run('{"prompt":null}') == ("null",)
    assert inputs["json_text"][1]["dynamicPrompts"] is False
    assert inputs["key_path"][1]["dynamicPrompts"] is False
