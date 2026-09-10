"""Pure image aspect-ratio selection helpers."""

from fractions import Fraction


ASPECT_RATIOS = (
    ("1:1", Fraction(1, 1)),
    ("16:9", Fraction(16, 9)),
    ("9/16", Fraction(9, 16)),
)


def closest_aspect_ratio(width: int, height: int) -> str:
    """Return the nearest width/height ratio, preferring square on ties."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    ratio = Fraction(width, height)
    return min(ASPECT_RATIOS, key=lambda preset: abs(ratio - preset[1]))[0]
