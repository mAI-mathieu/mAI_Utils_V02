import pytest

from utils.bounded_crop_geometry import bounded_crop_box


@pytest.mark.parametrize("bbox,expected", [
    ((400, 300, 500, 400), (194, 158, 512, 384)),
    ((0, 0, 30, 20), (0, 0, 512, 384)),
    ((970, 780, 1000, 800), (488, 416, 512, 384)),
    (None, (244, 208, 512, 384)),
])
def test_center_and_edges_preserve_entire_mask(bbox, expected):
    assert bounded_crop_box(1000, 800, bbox, 520, 390, 64) == expected


def test_source_smaller_than_limits_and_arbitrary_multiple():
    assert bounded_crop_box(110, 85, (80, 50, 100, 70), 200, 200, 12) == (2, 1, 108, 84)
    assert bounded_crop_box(7, 5, (6, 4, 7, 5), 100, 100, 1) == (0, 0, 7, 5)


@pytest.mark.parametrize("args,error", [
    ((20, 40, None, 64, 64, 32), "at least one"),
    ((100, 80, (0, 0, 100, 80), 100, 80, 16), "do not fit"),
    ((100, 80, (0, 0, 50, 50), 32, 64, 8), "do not fit"),
    ((100, 80, None, 32, 64, 0), "positive integer"),
    ((100, 80, None, 32, 64, 1.5), "positive integer"),
    ((100, 80, None, True, 64, 1), "positive integer"),
    ((100, 80, (-1, 0, 10, 10), 32, 64, 8), "inside the source"),
])
def test_impossible_requests_have_useful_errors(args, error):
    with pytest.raises(ValueError, match=error):
        bounded_crop_box(*args)


def test_geometry_invariants_across_sizes_and_edges():
    for image_width, image_height in ((35, 29), (200, 150), (1920, 1080)):
        for multiple in (1, 7, 16):
            for max_width, max_height in ((32, 32), (500, 300), (2048, 2048)):
                width = min(image_width, max_width) // multiple * multiple
                height = min(image_height, max_height) // multiple * multiple
                for left, top in ((0, 0), (image_width // 2, image_height // 2),
                                  (image_width - 1, image_height - 1)):
                    x, y, w, h = bounded_crop_box(
                        image_width, image_height, (left, top, left + 1, top + 1),
                        max_width, max_height, multiple,
                    )
                    assert (w, h) == (width, height)
                    assert w % multiple == h % multiple == 0
                    assert 0 <= x <= left < x + w <= image_width
                    assert 0 <= y <= top < y + h <= image_height
