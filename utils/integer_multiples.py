"""Exact integer rounding to multiples without floating-point conversion."""


def round_to_multiple(value: int, multiple: int, rounding: str = "round down") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("value must be an integer.")
    if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple <= 0:
        raise ValueError("multiple must be a positive integer.")
    if rounding == "round down":
        return (value // multiple) * multiple
    if rounding == "round up":
        return -(-value // multiple) * multiple
    raise ValueError(f"Unknown rounding mode: {rounding}")
