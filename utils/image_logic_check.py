"""Image shape measurements and numeric comparisons without tensor dependencies."""

import math


def image_property_value(shape, property):
    """Measure one property from a standard ComfyUI IMAGE batch shape."""
    if shape is None or len(shape) != 4 or any(size <= 0 for size in shape):
        raise ValueError(
            "Image must have non-empty shape [batch, height, width, channels]."
        )

    # ComfyUI IMAGE tensors use [B, H, W, C]; no pixel data is accessed.
    batch_size, height, width, _ = (int(size) for size in shape)
    if property == "Megapixels":
        # Decimal megapixels per image, independent of the batch size.
        return (width * height) / 1_000_000.0
    if property == "Width":
        return float(width)
    if property == "Height":
        return float(height)
    if property == "Aspect Ratio":
        return width / height
    if property == "Batch Size":
        return float(batch_size)
    raise ValueError(f"Unknown image property: {property}")


def compare_values(actual_value, operator, compare_value):
    """Compare numbers, using a small tolerance for equality and inequality."""
    if operator == ">":
        return actual_value > compare_value
    if operator == ">=":
        return actual_value >= compare_value
    if operator == "<":
        return actual_value < compare_value
    if operator == "<=":
        return actual_value <= compare_value
    if operator == "==":
        return math.isclose(actual_value, compare_value, rel_tol=1e-6, abs_tol=1e-6)
    if operator == "!=":
        return not math.isclose(actual_value, compare_value, rel_tol=1e-6, abs_tol=1e-6)
    raise ValueError(f"Unknown comparison operator: {operator}")
