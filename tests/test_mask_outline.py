import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from nodes.mask_outline import MAIMaskOutline
from utils.mask_outline import create_mask_outline, draw_mask_outline, parse_hex_color


def test_outline_follows_an_irregular_mask_instead_of_its_bounding_box():
    image = torch.zeros((1, 8, 8, 3))
    mask = torch.zeros((1, 8, 8))
    mask[0, 1:7, 1:4] = 1.0
    mask[0, 4:7, 1:7] = 1.0

    result = draw_mask_outline(image, mask, color="#00FF00", thickness=1)

    green = torch.tensor([0.0, 1.0, 0.0])
    assert torch.equal(result[0, 1, 1], green)
    assert torch.equal(result[0, 4, 6], green)
    assert torch.equal(result[0, 6, 6], green)
    assert torch.equal(result[0, 1, 6], torch.zeros(3))
    assert torch.equal(result[0, 3, 6], torch.zeros(3))


def test_one_pixel_outline_keeps_deep_mask_interior_unchanged():
    image = torch.zeros((1, 7, 7, 3))
    mask = torch.zeros((1, 7, 7))
    mask[0, 1:6, 1:6] = 1.0

    result = draw_mask_outline(image, mask, color="#FF0000", thickness=1)

    assert torch.equal(result[0, 1, 3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(result[0, 3, 3], torch.zeros(3))
    assert torch.equal(result[0, 0, 3], torch.zeros(3))


def test_even_thickness_is_split_inside_and_outside_the_mask_edge():
    mask = torch.zeros((1, 7, 7))
    mask[0, 2:5, 2:5] = 1.0

    outline = create_mask_outline(mask, thickness=2)

    assert outline[0, 1, 3]
    assert outline[0, 2, 3]
    assert outline[0, 4, 3]
    assert outline[0, 5, 3]
    assert not outline[0, 3, 3]
    assert not outline[0, 0, 3]


def test_padding_expands_the_mask_before_tracing_its_contour():
    mask = torch.zeros((1, 9, 9))
    mask[0, 4, 4] = 1.0

    outline = create_mask_outline(mask, thickness=1, padding=2)

    assert outline[0, 2, 4]
    assert outline[0, 6, 4]
    assert outline[0, 4, 2]
    assert outline[0, 4, 6]
    assert not outline[0, 4, 4]
    assert not outline[0, 1, 4]


def test_threshold_uses_zero_to_255_scale():
    image = torch.zeros((1, 5, 7, 3))
    mask = torch.zeros((1, 5, 7))
    mask[0, 0, 0] = 0.1
    mask[0, 2:4, 4:6] = 0.8

    result = draw_mask_outline(
        image,
        mask,
        color="#0000FF",
        thickness=1,
        threshold=128,
    )

    assert torch.equal(result[0, 0, 0], torch.zeros(3))
    assert torch.equal(result[0, 2, 4], torch.tensor([0.0, 0.0, 1.0]))


def test_empty_mask_returns_unchanged_image():
    image = torch.rand((1, 4, 5, 3))

    result = draw_mask_outline(image, torch.zeros((1, 4, 5)))

    assert torch.equal(result, image)
    assert result.data_ptr() != image.data_ptr()


def test_single_mask_broadcasts_over_image_batch_and_preserves_alpha():
    image = torch.zeros((2, 4, 5, 4))
    image[..., 3] = 0.25
    mask = torch.zeros((1, 4, 5))
    mask[0, 1:3, 1:4] = 1.0

    result = draw_mask_outline(image, mask, color="#FFFFFF", thickness=1)

    assert torch.equal(result[0], result[1])
    assert torch.all(result[..., 3] == 0.25)


def test_rejects_invalid_color_and_mismatched_dimensions():
    with pytest.raises(ValueError, match="#RRGGBB"):
        parse_hex_color("green")

    with pytest.raises(ValueError, match="dimensions must match"):
        draw_mask_outline(
            torch.zeros((1, 4, 5, 3)),
            torch.zeros((1, 5, 5)),
        )


def test_node_contract_defaults_and_output_tuple():
    inputs = MAIMaskOutline.INPUT_TYPES()["required"]
    assert list(inputs) == [
        "image",
        "mask",
        "color",
        "thickness",
        "padding",
        "threshold",
    ]
    assert inputs["color"][1]["default"] == "#00FF00"
    assert inputs["thickness"][1]["default"] == 10
    assert inputs["padding"][1]["default"] == 0
    assert inputs["threshold"][1]["default"] == 0
    assert MAIMaskOutline.CATEGORY == "mAI / Mask"
    assert MAIMaskOutline.RETURN_TYPES == ("IMAGE",)

    output = MAIMaskOutline().draw_outline(
        torch.zeros((1, 2, 2, 3)),
        torch.ones((1, 2, 2)),
        "#00FF00",
        1,
        0,
        0,
    )
    assert len(output) == 1


def test_whole_pack_imports_and_registers_node(monkeypatch):
    repo_dir = Path(__file__).resolve().parents[1]
    module_name = "test_mai_mask_outline_pack"
    spec = importlib.util.spec_from_file_location(
        module_name,
        repo_dir / "__init__.py",
        submodule_search_locations=[str(repo_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, package)
    spec.loader.exec_module(package)

    mapping_key = "MAIMaskOutline"
    assert package.NODE_CLASS_MAPPINGS[mapping_key].__name__ == mapping_key
    assert package.NODE_DISPLAY_NAME_MAPPINGS[mapping_key] == "mAI mask outline"
