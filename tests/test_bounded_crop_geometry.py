import pytest

from utils.bounded_crop_geometry import bounded_crop_box, bounded_crop_layout


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
    ((100, 80, None, 32, 64, 0), "positive integer"),
    ((100, 80, None, 32, 64, 1.5), "positive integer"),
    ((100, 80, None, True, 64, 1), "positive integer"),
    ((100, 80, (-1, 0, 10, 10), 32, 64, 8), "inside the source"),
])
def test_impossible_requests_have_useful_errors(args, error):
    with pytest.raises(ValueError, match=error):
        bounded_crop_box(*args)


def test_reported_1216_by_1288_mask_is_retained_and_downscaled():
    bbox = (100, 50, 1316, 1338)
    boxes, output_width, output_height = bounded_crop_layout(
        1400, 1450, [bbox], 1024, 1024, 64,
    )
    assert boxes == [(64, 50, 1288, 1288)]
    assert (output_width, output_height) == (1024, 1024)


def test_full_image_mask_can_exceed_aligned_output_without_padding():
    boxes, width, height = bounded_crop_layout(1216, 1288, [(0, 0, 1216, 1288)], 1024, 1024, 64)
    assert boxes == [(0, 0, 1216, 1288)]
    assert (width, height) == (960, 1024)


def test_oversize_batch_keeps_shared_dimensions_and_independent_positions():
    boxes, width, height = bounded_crop_layout(
        200, 150, [(0, 0, 10, 10), (100, 60, 200, 150)], 64, 64, 16,
    )
    assert boxes == [(0, 0, 100, 100), (100, 50, 100, 100)]
    assert (width, height) == (64, 64)


@pytest.mark.parametrize("image_size", [(35, 29), (200, 150), (1920, 1080)])
@pytest.mark.parametrize("multiple", [1, 7, 16])
def test_oversize_source_and_output_invariants(image_size, multiple):
    iw, ih = image_size
    bboxes = [(0, 0, iw, ih), (iw - 1, ih - 1, iw, ih)]
    for mw, mh in ((32, 32), (500, 300), (2048, 2048)):
        boxes, ow, oh = bounded_crop_layout(iw, ih, bboxes, mw, mh, multiple)
        assert 0 < ow <= min(iw, mw) and 0 < oh <= min(ih, mh)
        assert ow % multiple == oh % multiple == 0
        assert len({(box[2], box[3]) for box in boxes}) == 1
        for (x, y, w, h), (left, top, right, bottom) in zip(boxes, bboxes):
            assert 0 <= x <= left < right <= x + w <= iw
            assert 0 <= y <= top < bottom <= y + h <= ih
            assert ow <= w and oh <= h


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
