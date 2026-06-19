import pytest
import torch

from utils.image_composite import compose_layer_stack, composite_images


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


def test_fully_opaque_layer_replaces_background_in_overlap():
    background = rgb([0.0, 0.0, 0.0], 4, 4)
    foreground = rgb([1.0, 0.0, 0.0], 2, 2)

    result = composite_images(background, foreground, 1, 1, 1.0, "top_left", "none")

    assert_close(result[:, 1:3, 1:3, :], rgb([1.0, 0.0, 0.0], 2, 2))
    assert_close(result[:, 0:1, 0:1, :], rgb([0.0, 0.0, 0.0], 1, 1))


def test_fully_transparent_layer_leaves_background_unchanged():
    background = rgb([0.2, 0.4, 0.6], 3, 3)
    foreground = rgba([1.0, 0.0, 0.0], 0.0, 3, 3)

    result = composite_images(background, foreground, 0, 0, 1.0, "top_left", "none")

    assert_close(result, background)


def test_half_alpha_layer_blends_correctly():
    background = rgb([0.0, 0.0, 1.0], 1, 1)
    foreground = rgba([1.0, 0.0, 0.0], 0.5, 1, 1)

    result = composite_images(background, foreground, 0, 0, 1.0, "top_left", "none")

    assert_close(result, rgb([0.5, 0.0, 0.5], 1, 1))


def test_four_channel_layer_uses_embedded_alpha():
    background = rgb([0.0, 0.0, 0.0], 1, 1)
    foreground = rgba([0.0, 1.0, 0.0], 0.25, 1, 1)

    result = composite_images(background, foreground, 0, 0, 1.0, "top_left", "none")

    assert_close(result, rgb([0.0, 0.25, 0.0], 1, 1))


def test_layer_mask_overrides_embedded_alpha():
    background = rgb([0.0, 0.0, 0.0], 1, 1)
    foreground = rgba([1.0, 0.0, 0.0], 0.0, 1, 1)

    result = composite_images(
        background,
        foreground,
        0,
        0,
        1.0,
        "top_left",
        "none",
        foreground_mask=mask(1.0, 1, 1),
    )

    assert_close(result, rgb([1.0, 0.0, 0.0], 1, 1))


def test_layer_opacity_reduces_final_alpha():
    background = rgb([0.0, 0.0, 0.0], 1, 1)
    foreground = rgba([1.0, 0.0, 0.0], 1.0, 1, 1)

    result = composite_images(
        background,
        foreground,
        0,
        0,
        1.0,
        "top_left",
        "none",
        opacity=0.5,
    )

    assert_close(result, rgb([0.5, 0.0, 0.0], 1, 1))


def test_negative_x_and_y_crop_correctly():
    background = rgb([0.0, 0.0, 0.0], 3, 3)
    foreground = rgb([1.0, 0.0, 0.0], 2, 2)

    result = composite_images(background, foreground, -1, -1, 1.0, "top_left", "none")

    expected = background.clone()
    expected[:, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0])
    assert_close(result, expected)


def test_partly_outside_bottom_right_crops_correctly():
    background = rgb([0.0, 0.0, 0.0], 3, 3)
    foreground = rgb([1.0, 0.0, 0.0], 2, 2)

    result = composite_images(background, foreground, 2, 2, 1.0, "top_left", "none")

    expected = background.clone()
    expected[:, 2, 2, :] = torch.tensor([1.0, 0.0, 0.0])
    assert_close(result, expected)


def test_completely_outside_layer_returns_unchanged_background():
    background = rgb([0.2, 0.3, 0.4], 3, 3)
    foreground = rgb([1.0, 0.0, 0.0], 2, 2)

    result = composite_images(background, foreground, 4, 4, 1.0, "top_left", "none")

    assert_close(result, background)


def test_center_anchor_places_layer_around_xy_center():
    background = rgb([0.0, 0.0, 0.0], 4, 4)
    foreground = rgb([1.0, 0.0, 0.0], 2, 2)

    result = composite_images(background, foreground, 2, 2, 1.0, "center", "none")

    expected = background.clone()
    expected[:, 1:3, 1:3, :] = rgb([1.0, 0.0, 0.0], 2, 2)
    assert_close(result, expected)


def test_two_layers_stack_in_order():
    background = rgb([0.0, 0.0, 0.0], 2, 2)
    first = rgb([1.0, 0.0, 0.0], 2, 2)
    second = rgb([0.0, 1.0, 0.0], 1, 1)

    result = compose_layer_stack(
        background,
        [
            {"image": first, "x": 0, "y": 0, "scale": 1.0, "opacity": 1.0},
            {"image": second, "x": 0, "y": 0, "scale": 1.0, "opacity": 1.0},
        ],
    )

    assert_close(result[:, 0:1, 0:1, :], rgb([0.0, 1.0, 0.0], 1, 1))
    assert_close(result[:, 1:2, 1:2, :], rgb([1.0, 0.0, 0.0], 1, 1))


def test_empty_unused_layers_are_ignored():
    background = rgb([0.1, 0.2, 0.3], 2, 2)

    result = compose_layer_stack(background, [{}, {"image": None}])

    assert_close(result, background)


def test_background_batch_size_one_repeats_to_match_layer_batch():
    background = rgb([0.0, 0.0, 0.0], 1, 1, batch=1)
    foreground = torch.stack(
        [
            rgb([1.0, 0.0, 0.0], 1, 1)[0],
            rgb([0.0, 1.0, 0.0], 1, 1)[0],
        ],
        dim=0,
    )

    result = composite_images(background, foreground, 0, 0, 1.0, "top_left", "none")

    assert result.shape[0] == 2
    assert_close(result[0:1], rgb([1.0, 0.0, 0.0], 1, 1))
    assert_close(result[1:2], rgb([0.0, 1.0, 0.0], 1, 1))


def test_incompatible_batch_sizes_raise_value_error():
    background = rgb([0.0, 0.0, 0.0], 1, 1, batch=2)
    foreground = rgb([1.0, 0.0, 0.0], 1, 1, batch=3)

    with pytest.raises(ValueError):
        composite_images(background, foreground, 0, 0, 1.0, "top_left", "none")


def test_scale_less_than_or_equal_to_zero_raises_value_error():
    background = rgb([0.0, 0.0, 0.0], 1, 1)
    foreground = rgb([1.0, 0.0, 0.0], 1, 1)

    with pytest.raises(ValueError):
        composite_images(background, foreground, 0, 0, 0.0, "top_left", "none")


def test_contain_fit_mode_keeps_full_layer_visible_inside_background():
    background = rgb([0.0, 0.0, 0.0], 4, 4)
    foreground = rgb([1.0, 0.0, 0.0], 4, 2)

    result = composite_images(background, foreground, 0, 0, 1.0, "top_left", "contain")

    expected = background.clone()
    expected[:, 0:2, 0:4, :] = rgb([1.0, 0.0, 0.0], 4, 2)
    assert_close(result, expected)


def test_cover_fit_mode_covers_full_background():
    background = rgb([0.0, 0.0, 0.0], 4, 4)
    foreground = rgb([1.0, 0.0, 0.0], 4, 2)

    result = composite_images(background, foreground, 0, 0, 1.0, "top_left", "cover")

    assert_close(result, rgb([1.0, 0.0, 0.0], 4, 4))
