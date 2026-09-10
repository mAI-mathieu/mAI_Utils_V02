import pytest

from utils.aspect_ratios import closest_aspect_ratio


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (1024, 1024, "1:1"),
        (1920, 1080, "16:9"),
        (1080, 1920, "9/16"),
        (1200, 1000, "1:1"),
        (1500, 1000, "16:9"),
        (700, 1000, "9/16"),
        (800, 1000, "1:1"),
        (10000, 1, "16:9"),
        (1, 10000, "9/16"),
        (25, 18, "1:1"),  # Exact midpoint of 1:1 and 16:9.
        (25, 32, "1:1"),  # Exact midpoint of 1:1 and 9/16.
    ],
)
def test_closest_aspect_ratio(width, height, expected):
    assert closest_aspect_ratio(width, height) == expected


@pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_rejects_nonpositive_dimensions(width, height):
    with pytest.raises(ValueError, match="must be positive"):
        closest_aspect_ratio(width, height)
