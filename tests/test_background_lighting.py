import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from nodes.background_lighting_match import MAIBackgroundLightingMatch
from utils.background_lighting import match_background_lighting


def test_restores_affine_brightness_and_contrast_shift():
    original = torch.tensor(
        [
            [
                [[0.10, 0.20, 0.30], [0.20, 0.30, 0.40]],
                [[0.30, 0.40, 0.50], [0.40, 0.50, 0.60]],
            ]
        ],
        dtype=torch.float32,
    )
    edited = original * 0.5 + 0.2
    edit_mask = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])

    result = match_background_lighting(original, edited, edit_mask)

    assert torch.allclose(result, original, atol=1e-6)


def test_masked_pixels_do_not_influence_the_reference_statistics():
    original = torch.tensor(
        [[[[0.20, 0.30, 0.40], [0.40, 0.50, 0.60], [1.0, 0.0, 1.0]]]]
    )
    edited = torch.tensor(
        [[[[0.30, 0.35, 0.40], [0.50, 0.45, 0.50], [0.90, 0.90, 0.10]]]]
    )
    edit_mask = torch.tensor([[[0.0, 0.0, 1.0]]])

    result = match_background_lighting(original, edited, edit_mask)

    expected_masked_pixel = torch.tensor([0.80, 1.00, 0.00])
    assert torch.allclose(result[0, 0, :2], original[0, 0, :2], atol=1e-6)
    assert torch.allclose(result[0, 0, 2], expected_masked_pixel, atol=1e-6)


def test_single_original_and_mask_broadcast_across_edited_batch():
    original = torch.tensor(
        [[[[0.20, 0.20, 0.20], [0.60, 0.60, 0.60]]]], dtype=torch.float32
    )
    edited = torch.cat((original * 0.5 + 0.1, original * 0.25 + 0.3), dim=0)
    edit_mask = torch.zeros((1, 1, 2), dtype=torch.float32)

    result = match_background_lighting(original, edited, edit_mask)

    assert result.shape[0] == 2
    assert torch.allclose(result[0], original[0], atol=1e-6)
    assert torch.allclose(result[1], original[0], atol=1e-6)


def test_soft_mask_weights_background_statistics_proportionally():
    original = torch.tensor([[[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]]])
    edited = original * 0.5 + 0.2
    edit_mask = torch.tensor([[[0.0, 0.5, 1.0]]])

    result = match_background_lighting(original, edited, edit_mask)

    assert torch.allclose(result, original, atol=1e-6)


def test_fully_white_mask_raises_clear_error():
    image = torch.full((1, 2, 2, 3), 0.5)
    edit_mask = torch.ones((1, 2, 2))

    with pytest.raises(ValueError, match="no untouched background"):
        match_background_lighting(image, image, edit_mask)


def test_mismatched_image_dimensions_raise_clear_error():
    original = torch.zeros((1, 2, 2, 3))
    edited = torch.zeros((1, 3, 2, 3))
    edit_mask = torch.zeros((1, 3, 2))

    with pytest.raises(ValueError, match="dimensions must match"):
        match_background_lighting(original, edited, edit_mask)


def test_unrecoverable_zero_contrast_raises_clear_error():
    original = torch.tensor(
        [[[[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]]], dtype=torch.float32
    )
    edited = torch.full_like(original, 0.5)
    edit_mask = torch.zeros((1, 1, 2))

    with pytest.raises(ValueError, match="zero contrast"):
        match_background_lighting(original, edited, edit_mask)


def test_node_contract_and_output_tuple():
    node = MAIBackgroundLightingMatch()
    original = torch.tensor(
        [[[[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]]]], dtype=torch.float32
    )
    edited = original * 0.5 + 0.1
    edit_mask = torch.zeros((1, 1, 2))

    output = node.match_lighting(original, edited, edit_mask)

    assert MAIBackgroundLightingMatch.CATEGORY == "mAI / Image"
    assert MAIBackgroundLightingMatch.RETURN_TYPES == ("IMAGE",)
    assert len(output) == 1
    assert torch.allclose(output[0], original, atol=1e-6)


def test_whole_pack_imports_and_registers_node(monkeypatch):
    repo_dir = Path(__file__).resolve().parents[1]
    module_name = "test_mai_background_lighting_pack"
    spec = importlib.util.spec_from_file_location(
        module_name,
        repo_dir / "__init__.py",
        submodule_search_locations=[str(repo_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, package)
    spec.loader.exec_module(package)

    mapping_key = "MAIBackgroundLightingMatch"
    assert package.NODE_CLASS_MAPPINGS[mapping_key].__name__ == mapping_key
    assert package.NODE_DISPLAY_NAME_MAPPINGS[mapping_key] == (
        "mAI background lighting match"
    )
