import random

import pytest

from utils.mask_crop_geometry import calculate_crop_geometry, valid_crop_bounds


@pytest.mark.parametrize("image_size,bbox,target,expected", [
    ((2048, 2048), (874, 824, 1174, 1224), (1024, 1024), (512, 512, 1024, 1024, 1.0)),
    ((2048, 2048), (0, 100, 300, 500), (1024, 1024), (0, 0, 1024, 1024, 1.0)),
    ((3000, 2000), (800, 650, 2200, 1350), (1024, 1024), (800, 300, 1400, 1400, 1024 / 1400)),
    ((3000, 2000), (1300, 850, 1700, 1150), (1280, 768), (860, 616, 1280, 768, 1.0)),
    ((800, 600), None, (1024, 1024), (-112, -212, 1024, 1024, 1.0)),
])
def test_requested_acceptance_geometry(image_size, bbox, target, expected):
    item = calculate_crop_geometry(*image_size, bbox, *target)
    assert tuple(item[k] for k in ("crop_x", "crop_y", "source_crop_width", "source_crop_height", "scale")) == expected
    assert item["was_resized"] == (expected[-1] != 1)


def test_padding_threshold_geometry_and_deterministic_odd_distribution():
    item = calculate_crop_geometry(100, 100, (20, 30, 23, 35), 6, 8, edge_mode="pad")
    # Three extra pixels: one before, two after, on each axis.
    assert (item["crop_x"], item["crop_y"]) == (19, 29)
    padded = calculate_crop_geometry(100, 100, (0, 1, 10, 9), 8, 8, mask_padding=5)
    assert padded["padded_mask_bbox"] == (0, 0, 15, 14)
    assert (padded["source_crop_width"], padded["source_crop_height"]) == (15, 15)


def test_pad_preserves_center_and_shift_uses_image_pixels():
    pad = calculate_crop_geometry(20, 20, (0, 0, 2, 2), 10, 10, edge_mode="pad")
    shift = calculate_crop_geometry(20, 20, (0, 0, 2, 2), 10, 10)
    assert (pad["crop_x"], pad["crop_y"], pad["pad_left"], pad["pad_top"]) == (-4, -4, 4, 4)
    assert valid_crop_bounds(pad) == (0, 0, 6, 6)
    assert valid_crop_bounds(shift) == (0, 0, 10, 10)
    assert not shift["was_padded"]


def test_upscale_is_opt_in_and_never_loses_the_mask():
    native = calculate_crop_geometry(800, 600, (390, 290, 410, 310), 1024, 1024)
    upscale = calculate_crop_geometry(800, 600, (390, 290, 410, 310), 1024, 1024, allow_upscale=True)
    assert native["scale"] == 1
    assert upscale["source_crop_width"] == upscale["source_crop_height"] == 600
    assert upscale["scale"] == 1024 / 600
    assert not upscale["was_padded"]
    full = calculate_crop_geometry(800, 600, (0, 0, 800, 600), 1024, 1024, allow_upscale=True)
    assert full["scale"] == 1  # Square 600 crop cannot contain an 800-wide mask.


def test_rectangular_downscale_uses_uniform_scale_with_integer_geometry():
    item = calculate_crop_geometry(3000, 2000, (400, 200, 1801, 1000), 1280, 768)
    assert (item["source_crop_width"], item["source_crop_height"]) == (1405, 843)
    assert item["scale"] == 1280 / 1405 == 768 / 843
    odd = calculate_crop_geometry(30, 30, (2, 3, 20, 21), 13, 7)
    assert (odd["source_crop_width"], odd["source_crop_height"]) == (39, 21)


def test_geometry_invariants_across_many_sizes_edges_and_aspect_ratios():
    rng = random.Random(731)
    for _ in range(400):
        width, height, ow, oh = [rng.randint(1, 150) for _ in range(4)]
        left, top = rng.randrange(width), rng.randrange(height)
        bbox = (left, top, rng.randint(left + 1, width), rng.randint(top + 1, height))
        item = calculate_crop_geometry(width, height, bbox, ow, oh,
                                       rng.randint(0, 15), rng.choice(("shift", "pad")), rng.choice((False, True)))
        x, y, sw, sh = [item[k] for k in ("crop_x", "crop_y", "source_crop_width", "source_crop_height")]
        l, t, r, b = item["padded_mask_bbox"]
        assert x <= l < r <= x + sw
        assert y <= t < b <= y + sh
        assert sw * oh == sh * ow
        assert item["scale"] == pytest.approx(ow / sw)
        assert item["scale"] == pytest.approx(oh / sh)
        vl, vt, vr, vb = valid_crop_bounds(item)
        assert sw == item["pad_left"] + vr - vl + item["pad_right"]
        assert sh == item["pad_top"] + vb - vt + item["pad_bottom"]


@pytest.mark.parametrize("kwargs,error", [
    ({"output_width": 0}, "output_width"),
    ({"output_height": 1.5}, "output_height"),
    ({"mask_padding": -1}, "mask_padding"),
    ({"edge_mode": "unknown"}, "edge_mode"),
    ({"allow_upscale": 1}, "allow_upscale"),
    ({"mask_bbox": (0, 0, 11, 10)}, "mask_bbox"),
])
def test_bad_settings_report_clear_errors(kwargs, error):
    settings = dict(original_width=10, original_height=10, mask_bbox=None,
                    output_width=5, output_height=5)
    settings.update(kwargs)
    with pytest.raises(ValueError, match=error):
        calculate_crop_geometry(**settings)
