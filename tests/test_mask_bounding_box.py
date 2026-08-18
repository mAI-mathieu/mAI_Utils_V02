import pytest
import torch

from utils.mask_bounding_box import create_bounding_box_mask


def test_creates_filled_rectangle_around_nonzero_pixels():
    source = torch.zeros((1, 6, 7), dtype=torch.float32)
    source[0, 1, 2] = 0.25
    source[0, 4, 5] = 1.0

    result = create_bounding_box_mask(source)

    expected = torch.zeros_like(source)
    expected[0, 1:5, 2:6] = 1.0
    assert torch.equal(result, expected)


def test_creates_a_separate_rectangle_for_each_batch_item():
    source = torch.zeros((2, 5, 6), dtype=torch.float32)
    source[0, 0, 1] = 1.0
    source[0, 2, 3] = 1.0
    source[1, 3, 4] = 1.0

    result = create_bounding_box_mask(source)

    expected = torch.zeros_like(source)
    expected[0, 0:3, 1:4] = 1.0
    expected[1, 3, 4] = 1.0
    assert torch.equal(result, expected)


def test_empty_mask_remains_empty():
    source = torch.zeros((1, 4, 4), dtype=torch.float32)

    result = create_bounding_box_mask(source)

    assert torch.equal(result, source)


def test_supports_a_two_dimensional_mask():
    source = torch.zeros((4, 5), dtype=torch.float32)
    source[1:3, 2] = 1.0

    result = create_bounding_box_mask(source)

    expected = torch.zeros_like(source)
    expected[1:3, 2] = 1.0
    assert torch.equal(result, expected)


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="mask must have shape"):
        create_bounding_box_mask(torch.zeros((1, 2, 3, 1)))
