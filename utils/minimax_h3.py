import math
import operator


DIMENSION_MULTIPLE = 32
PIXELS_PER_MEGAPIXEL = 1024 * 1024
MEGAPIXEL_OPTIONS = tuple(f"{value / 10:.1f} MP" for value in range(2, 21))
RESIZE_MODE_MEGAPIXELS = "Target megapixels"
RESIZE_MODE_SHORT_SIDE = "Short side"
# Accepted by the backend for older saved/API workflows.
RESIZE_MODE_SHORT_SIDE_768 = "Short side 768 px"
RESIZE_MODE_OPTIONS = (RESIZE_MODE_MEGAPIXELS, RESIZE_MODE_SHORT_SIDE)
RESIZE_ALGORITHM_OPTIONS = ("lanczos", "bicubic", "bilinear", "nearest-exact", "area")
SHORT_SIDE_TARGET = 768


def parse_target_megapixels(value):
    if isinstance(value, str):
        value = value.strip()
        if value.lower().endswith("mp"):
            value = value[:-2].strip()

    try:
        megapixels = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid target megapixel value: {value}") from error

    valid_values = {option / 10 for option in range(2, 21)}
    if megapixels not in valid_values:
        raise ValueError("target_megapixels must be from 0.2 MP to 2.0 MP in 0.1 MP steps")

    return megapixels


def _positive_integer(value, name):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if isinstance(value, bool) or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def calculate_minimax_h3_dimensions(
    width, height, target_megapixels, multiple_of=DIMENSION_MULTIPLE
):
    """Return proportional target dimensions rounded to the selected pixel grid."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be greater than 0")
    multiple_of = _positive_integer(multiple_of, "multiple_of")

    megapixels = parse_target_megapixels(target_megapixels)
    target_pixels = megapixels * PIXELS_PER_MEGAPIXEL
    scale = math.sqrt(target_pixels / (width * height))

    ideal_width = width * scale
    ideal_height = height * scale

    target_width = max(
        multiple_of,
        int(math.floor(ideal_width / multiple_of + 0.5)) * multiple_of,
    )
    target_height = max(
        multiple_of,
        int(math.floor(ideal_height / multiple_of + 0.5)) * multiple_of,
    )

    return target_width, target_height


def calculate_short_side_dimensions(
    width, height, short_side=SHORT_SIDE_TARGET, multiple_of=DIMENSION_MULTIPLE
):
    """Scale to a short-side target, then round both dimensions to the pixel grid."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be greater than 0")
    short_side = _positive_integer(short_side, "short_side")
    multiple_of = _positive_integer(multiple_of, "multiple_of")

    scale = short_side / min(width, height)
    ideal_width = width * scale
    ideal_height = height * scale

    target_width = max(
        multiple_of,
        int(math.floor(ideal_width / multiple_of + 0.5)) * multiple_of,
    )
    target_height = max(
        multiple_of,
        int(math.floor(ideal_height / multiple_of + 0.5)) * multiple_of,
    )

    return target_width, target_height
