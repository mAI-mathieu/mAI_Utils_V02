import pytest
import torch

from utils.image_composite import composite_layer


def rgb(color, width, height, batch=1):
    tensor = torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3)
    return tensor.repeat(batch, height, width, 1)


def rgba(color, alpha, width, height, batch=1):
    tensor = torch.tensor([*color, alpha], dtype=torch.float32).view(1, 1, 1, 4)
    return tensor.repeat(batch, height, width, 1)


def mask(value, width, height, batch=1):
    return torch.full((batch, height, width), value, dtype=torch.float32)


def assert_close(actual, expected):
    assert torch.allclose(actual, expected, atol=1e-6)


def compose(base, layer, **kwargs):
    defaults = {
        "x": 0,
        "y": 0,
        "scale": 1.0,
        "opacity": 1.0,
        "anchor": "top_left",
        "fit_mode": "none",
    }
    defaults.update(kwargs)
    return composite_layer(base, layer, **defaults)


def test_opaque_layer_replaces_base_in_overlap():
    base = rgb([0.0, 0.0, 0.0], 4, 4)
    layer = rgb([1.0, 0.0, 0.0], 2, 2)

    image, output_mask = compose(base, layer, x=1, y=1)

    assert_close(image[:, 1:3, 1:3, :], rgb([1.0, 0.0, 0.0], 2, 2))
    assert_close(image[:, 0:1, 0:1, :], rgb([0.0, 0.0, 0.0], 1, 1))
    assert_close(output_mask[:, 1:3, 1:3], mask(1.0, 2, 2))


def test_transparent_layer_leaves_base_unchanged():
    base = rgb([0.2, 0.4, 0.6], 3, 3)
    layer = rgba([1.0, 0.0, 0.0], 0.0, 3, 3)

    image, output_mask = compose(base, layer)

    assert_close(image, base)
    assert_close(output_mask, mask(0.0, 3, 3))


def test_half_alpha_blends_correctly():
    base = rgb([0.0, 0.0, 1.0], 1, 1)
    layer = rgba([1.0, 0.0, 0.0], 0.5, 1, 1)

    image, _output_mask = compose(base, layer)

    assert_close(image, rgb([0.5, 0.0, 0.5], 1, 1))


def test_four_channel_layer_image_uses_embedded_alpha():
    base = rgb([0.0, 0.0, 0.0], 1, 1)
    layer = rgba([0.0, 1.0, 0.0], 0.25, 1, 1)

    image, output_mask = compose(base, layer)

    assert_close(image, rgb([0.0, 0.25, 0.0], 1, 1))
    assert_close(output_mask, mask(0.25, 1, 1))


def test_layer_mask_overrides_embedded_alpha():
    base = rgb([0.0, 0.0, 0.0], 1, 1)
    layer = rgba([1.0, 0.0, 0.0], 0.0, 1, 1)

    image, output_mask = compose(base, layer, layer_mask=mask(1.0, 1, 1))

    assert_close(image, rgb([1.0, 0.0, 0.0], 1, 1))
    assert_close(output_mask, mask(1.0, 1, 1))


def test_opacity_reduces_final_alpha():
    base = rgb([0.0, 0.0, 0.0], 1, 1)
    layer = rgba([1.0, 0.0, 0.0], 1.0, 1, 1)

    image, output_mask = compose(base, layer, opacity=0.5)

    assert_close(image, rgb([0.5, 0.0, 0.0], 1, 1))
    assert_close(output_mask, mask(0.5, 1, 1))


def test_negative_x_and_y_crops_correctly():
    base = rgb([0.0, 0.0, 0.0], 3, 3)
    layer = rgb([1.0, 0.0, 0.0], 2, 2)

    image, output_mask = compose(base, layer, x=-1, y=-1)

    expected = base.clone()
    expected[:, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0])
    expected_mask = mask(0.0, 3, 3)
    expected_mask[:, 0, 0] = 1.0
    assert_close(image, expected)
    assert_close(output_mask, expected_mask)


def test_layer_partly_outside_bottom_right_crops_correctly():
    base = rgb([0.0, 0.0, 0.0], 3, 3)
    layer = rgb([1.0, 0.0, 0.0], 2, 2)

    image, output_mask = compose(base, layer, x=2, y=2)

    expected = base.clone()
    expected[:, 2, 2, :] = torch.tensor([1.0, 0.0, 0.0])
    expected_mask = mask(0.0, 3, 3)
    expected_mask[:, 2, 2] = 1.0
    assert_close(image, expected)
    assert_close(output_mask, expected_mask)


def test_layer_fully_outside_returns_unchanged_image():
    base = rgb([0.2, 0.3, 0.4], 3, 3)
    layer = rgb([1.0, 0.0, 0.0], 2, 2)

    image, output_mask = compose(base, layer, x=4, y=4)

    assert_close(image, base)
    assert_close(output_mask, mask(0.0, 3, 3))


def test_center_anchor_places_layer_around_xy_center():
    base = rgb([0.0, 0.0, 0.0], 4, 4)
    layer = rgb([1.0, 0.0, 0.0], 2, 2)

    image, output_mask = compose(base, layer, x=2, y=2, anchor="center")

    expected = base.clone()
    expected[:, 1:3, 1:3, :] = rgb([1.0, 0.0, 0.0], 2, 2)
    expected_mask = mask(0.0, 4, 4)
    expected_mask[:, 1:3, 1:3] = 1.0
    assert_close(image, expected)
    assert_close(output_mask, expected_mask)


def test_output_mask_matches_placed_layer_alpha_without_base_mask():
    base = rgb([0.0, 0.0, 0.0], 3, 3)
    layer = rgba([1.0, 0.0, 0.0], 0.25, 1, 1)

    _image, output_mask = compose(base, layer, x=1, y=1)

    expected_mask = mask(0.0, 3, 3)
    expected_mask[:, 1, 1] = 0.25
    assert_close(output_mask, expected_mask)


def test_output_mask_alpha_over_combines_with_base_mask():
    base = rgb([0.0, 0.0, 0.0], 3, 3)
    layer = rgba([1.0, 0.0, 0.0], 0.5, 1, 1)
    base_mask = mask(0.25, 3, 3)

    _image, output_mask = compose(base, layer, x=1, y=1, base_mask=base_mask)

    expected_mask = base_mask.clone()
    expected_mask[:, 1, 1] = 0.5 + 0.25 * (1.0 - 0.5)
    assert_close(output_mask, expected_mask)


def test_two_chained_calls_produce_expected_image_order():
    base = rgb([0.0, 0.0, 0.0], 1, 1)
    first = rgb([1.0, 0.0, 0.0], 1, 1)
    second = rgb([0.0, 1.0, 0.0], 1, 1)

    first_image, first_mask = compose(base, first)
    final_image, _final_mask = compose(first_image, second, base_mask=first_mask)

    assert_close(final_image, rgb([0.0, 1.0, 0.0], 1, 1))


def test_two_chained_calls_produce_expected_combined_mask():
    base = rgb([0.0, 0.0, 0.0], 1, 1)
    first = rgba([1.0, 0.0, 0.0], 0.5, 1, 1)
    second = rgba([0.0, 1.0, 0.0], 0.5, 1, 1)

    first_image, first_mask = compose(base, first)
    _final_image, final_mask = compose(first_image, second, base_mask=first_mask)

    assert_close(final_mask, mask(0.75, 1, 1))


def test_contain_fit_mode_keeps_full_layer_visible():
    base = rgb([0.0, 0.0, 0.0], 4, 4)
    layer = rgb([1.0, 0.0, 0.0], 4, 2)

    image, output_mask = compose(base, layer, fit_mode="contain")

    expected = base.clone()
    expected[:, 0:2, 0:4, :] = rgb([1.0, 0.0, 0.0], 4, 2)
    expected_mask = mask(0.0, 4, 4)
    expected_mask[:, 0:2, 0:4] = 1.0
    assert_close(image, expected)
    assert_close(output_mask, expected_mask)


def test_cover_fit_mode_covers_the_base_image():
    base = rgb([0.0, 0.0, 0.0], 4, 4)
    layer = rgb([1.0, 0.0, 0.0], 4, 2)

    image, output_mask = compose(base, layer, fit_mode="cover")

    assert_close(image, rgb([1.0, 0.0, 0.0], 4, 4))
    assert_close(output_mask, mask(1.0, 4, 4))


def test_incompatible_batch_sizes_raise_value_error():
    base = rgb([0.0, 0.0, 0.0], 1, 1, batch=2)
    layer = rgb([1.0, 0.0, 0.0], 1, 1, batch=3)

    with pytest.raises(ValueError):
        compose(base, layer)


def test_scale_less_than_or_equal_to_zero_raises_value_error():
    base = rgb([0.0, 0.0, 0.0], 1, 1)
    layer = rgb([1.0, 0.0, 0.0], 1, 1)

    with pytest.raises(ValueError):
        compose(base, layer, scale=0.0)
