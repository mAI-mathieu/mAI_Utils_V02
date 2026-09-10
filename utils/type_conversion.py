"""Scalar conversions independent of ComfyUI and its frontend."""

OUTPUT_TYPES = ("string", "int", "float", "boolean")


def convert_value(value, output_type="string", strict=False):
    """Detect the source from its Python value and convert only the chosen type."""
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"Unknown output_type: {output_type}")
    if not isinstance(value, (str, bool, int, float)):
        raise ValueError(
            "Type converter input must be a string, int, float, or boolean; "
            f"received {type(value).__name__}."
        )

    if output_type == "string":
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    try:
        if output_type == "boolean":
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"true", "1", "yes", "y", "on"}:
                    return True
                if text in {"false", "0", "no", "n", "off", ""}:
                    return False
                return float(text) != 0.0
            return value != 0

        if output_type == "int":
            if isinstance(value, str):
                text = value.strip()
                try:
                    # Preserve precision for integer strings, including seeds.
                    return int(text)
                except ValueError:
                    return int(float(text))
            return int(value)

        return float(value)
    except (ValueError, OverflowError) as exc:
        if strict:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to {output_type}: {value!r}"
            ) from exc
        return {"boolean": False, "int": 0, "float": 0.0}[output_type]
