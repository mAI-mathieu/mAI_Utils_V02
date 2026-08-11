import math


DIMENSION_MULTIPLE = 32
PIXELS_PER_MEGAPIXEL = 1024 * 1024
MEGAPIXEL_OPTIONS = tuple(f"{value / 10:.1f} MP" for value in range(2, 21))


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


def calculate_minimax_h3_dimensions(width, height, target_megapixels):
    """Return the closest proportional target dimensions on a 32-pixel grid."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be greater than 0")

    megapixels = parse_target_megapixels(target_megapixels)
    target_pixels = megapixels * PIXELS_PER_MEGAPIXEL
    scale = math.sqrt(target_pixels / (width * height))

    ideal_width = width * scale
    ideal_height = height * scale

    target_width = max(
        DIMENSION_MULTIPLE,
        int(math.floor(ideal_width / DIMENSION_MULTIPLE + 0.5)) * DIMENSION_MULTIPLE,
    )
    target_height = max(
        DIMENSION_MULTIPLE,
        int(math.floor(ideal_height / DIMENSION_MULTIPLE + 0.5)) * DIMENSION_MULTIPLE,
    )

    return target_width, target_height
