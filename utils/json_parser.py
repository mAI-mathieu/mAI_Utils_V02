"""Parse JSON and select one value without depending on ComfyUI."""

import json
import math
import re


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number is outside the supported finite float range.")
    return number


def parse_json_path(key_path):
    """Return object keys and nonnegative array indices; blank or $ means root.

    Dot notation selects object keys. Brackets accept array indices or JSON
    double-quoted keys, including keys containing dots, spaces, or brackets.
    This is a selector, not a JSONPath expression evaluator.
    """
    if not isinstance(key_path, str):
        raise ValueError("JSON key_path must be text.")
    path = key_path.strip()
    if not path or path == "$":
        return []
    position = 0
    if path.startswith("$"):
        position = 1
        if path[position] not in ".[":
            raise ValueError('After $, use a dot or brackets; quote literal keys with ["key"].')

    tokens = []
    decoder = json.JSONDecoder()
    while position < len(path):
        char = path[position]
        if char == "[":
            position += 1
            while position < len(path) and path[position].isspace():
                position += 1
            if position < len(path) and path[position] == '"':
                try:
                    token, position = decoder.raw_decode(path, position)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid quoted key in JSON path at character {position + 1}.") from exc
            else:
                match = re.match(r"[0-9]+", path[position:])
                if match is None:
                    raise ValueError("JSON path brackets require a nonnegative index or a double-quoted key.")
                token = int(match.group())
                position += len(match.group())
            while position < len(path) and path[position].isspace():
                position += 1
            if position >= len(path) or path[position] != "]":
                raise ValueError("Missing closing ] in JSON path.")
            position += 1
        else:
            if char == ".":
                if position == 0:
                    raise ValueError("JSON path cannot start with a dot; use $.key or key.")
                position += 1
            elif position != 0:
                raise ValueError(f"Expected a dot or brackets in JSON path at character {position + 1}.")
            start = position
            while position < len(path) and path[position] not in ".[]":
                position += 1
            token = path[start:position]
            if not token or token != token.strip():
                raise ValueError('Invalid JSON path key; use ["key"] for empty keys or surrounding spaces.')
        tokens.append(token)
    return tokens


def select_json_value(json_text, key_path="", output_type="auto"):
    """Return a scalar, or JSON text for objects, arrays and null.

    Null becomes the string 'null', so it never unexpectedly supplies None to a
    downstream ComfyUI node. String mode serializes numbers/booleans as JSON,
    while leaving selected strings unquoted.
    """
    if output_type not in ("auto", "string"):
        raise ValueError(f"Unknown JSON output type: {output_type}")
    if not isinstance(json_text, str) or not json_text.strip():
        raise ValueError("JSON input must be nonempty text.")
    try:
        value = json.loads(json_text, parse_constant=_reject_constant, parse_float=_finite_float)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc

    for token in parse_json_path(key_path):
        if isinstance(token, int):
            if not isinstance(value, list):
                raise ValueError(f"JSON path {key_path!r}: index [{token}] requires an array.")
            if token >= len(value):
                raise ValueError(f"JSON path {key_path!r}: array index {token} is out of range (length {len(value)}).")
        else:
            if not isinstance(value, dict):
                raise ValueError(f"JSON path {key_path!r}: key {token!r} requires an object.")
            if token not in value:
                raise ValueError(f"JSON path {key_path!r}: key {token!r} was not found.")
        value = value[token]

    if isinstance(value, str):
        return value
    if output_type == "string" or value is None or isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value
