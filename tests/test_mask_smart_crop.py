import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from nodes.mask_smart_crop import MAIMaskSmartCrop
from utils.mask_crop_geometry import valid_crop_bounds
from utils.mask_smart_crop import CONTEXT_TYPE, smart_crop


@pytest.mark.parametrize("width,height,bbox,ow,oh", [
    (2048, 2048, (874, 824, 1174, 1224), 1024, 1024),
    (2048, 2048, (0, 100, 300, 500), 1024, 1024),
    (3000, 2000, (800, 650, 2200, 1350), 1024, 1024),
    (3000, 2000, (1300, 850, 1700, 1150), 1280, 768),
    (800, 600, (390, 290, 410, 310), 1024, 1024),
])
def test_acceptance_sizes_and_full_crop_mask_consistency(width, height, bbox, ow, oh):
    image = torch.linspace(0, 1, width * height).reshape(1, height, width, 1)
    mask = torch.zeros(1, height, width)
    l, t, r, b = bbox
    mask[:, t:b, l:r] = 1
    crop, cropped_mask, full_mask, context, x, y, sw, sh, scale = smart_crop(image, mask, ow, oh)
    assert crop.shape == (1, oh, ow, 1)
    assert cropped_mask.shape == (1, oh, ow)
    assert full_mask.shape == (1, height, width)
    assert (x, y, sw, sh, scale) == tuple(context[k] for k in ("crop_x", "crop_y", "source_crop_width", "source_crop_height", "scale"))
    vl, vt, vr, vb = valid_crop_bounds(context)
    expected = torch.zeros_like(full_mask)
    expected[:, vt:vb, vl:vr] = 1
    assert torch.equal(full_mask, expected)
    assert torch.all(full_mask[mask > 0] == 1)
    if scale == 1:
        assert torch.equal(crop[:, vt-y:vb-y, vl-x:vr-x], image[:, vt:vb, vl:vr])
        assert torch.equal(cropped_mask[:, vt-y:vb-y, vl-x:vr-x], mask[:, vt:vb, vl:vr])


def test_empty_mask_center_threshold_and_soft_values_are_retained():
    image = torch.rand(1, 9, 11, 3)
    mask = torch.zeros(9, 11)
    empty = smart_crop(image, mask, 5, 5)
    assert empty[4:6] == (3, 2)
    assert empty[3]["mask_bbox"] is None
    mask[3, 4] = 0.01
    mask[5, 6] = 0.3
    cropped = smart_crop(image, mask, 5, 5)
    assert cropped[3]["mask_bbox"] == (6, 5, 7, 6)
    assert cropped[1].max() == 0.3
    # Context remains a native mask snapshot if upstream tensors are changed.
    saved = cropped[3]["source_mask"].clone()
    mask.fill_(1)
    assert torch.equal(cropped[3]["source_mask"], saved)


def test_different_batch_crops_and_single_mask_broadcast():
    image = torch.rand(2, 12, 16, 3)
    masks = torch.zeros(2, 12, 16)
    masks[0, 1, 1] = 1
    masks[1, 10, 14] = 1
    cropped = smart_crop(image, masks, 6, 6)
    assert cropped[3]["batch_size"] == 2
    assert [(i["crop_x"], i["crop_y"]) for i in cropped[3]["items"]] == [(0, 0), (10, 6)]
    assert not torch.equal(cropped[2][0], cropped[2][1])
    broadcast = smart_crop(image, masks[:1], 6, 6)
    assert torch.equal(broadcast[1][0], broadcast[1][1])
    assert torch.equal(broadcast[2][0], broadcast[2][1])


def test_pad_replicates_image_edges_and_zeros_mask_padding_even_for_tiny_source():
    image = torch.tensor([[[[0.3, 0.7, 1.0]]]])
    mask = torch.ones(1, 1, 1)
    crop, cropped_mask, full_mask, context, *_ = smart_crop(image, mask, 7, 5, edge_mode="pad")
    assert torch.equal(crop, image.expand(1, 5, 7, 3))
    assert cropped_mask.sum() == full_mask.sum() == 1
    assert context["pad_left"] == context["pad_right"] == 3
    assert context["pad_top"] == context["pad_bottom"] == 2


def test_mixed_image_mask_precision_does_not_quantize_bbox_threshold():
    image = torch.zeros(1, 9, 9, 3, dtype=torch.float16)
    mask = torch.zeros(9, 9, dtype=torch.float32)
    mask[1, 1] = 0.010001
    cropped = smart_crop(image, mask, 5, 5)
    assert cropped[3]["mask_bbox"] == (1, 1, 2, 2)
    assert cropped[0].dtype == image.dtype
    assert cropped[1].dtype == cropped[2].dtype == mask.dtype


def test_node_contract():
    node = MAIMaskSmartCrop()
    assert node.CATEGORY == "mAI / Mask"
    assert node.RETURN_TYPES == ("IMAGE", "MASK", "MASK", CONTEXT_TYPE, "INT", "INT", "INT", "INT", "FLOAT")
    assert node.RETURN_NAMES == ("cropped_image", "cropped_mask", "full_crop_mask", "crop_context", "crop_x", "crop_y", "crop_width", "crop_height", "scale")
    inputs = node.INPUT_TYPES()
    assert inputs["required"]["output_width"][1]["default"] == 1024
    assert inputs["optional"]["allow_upscale"][1]["default"] is False
    assert inputs["optional"]["edge_mode"][1]["default"] == "shift"


def test_pack_import_and_registration_preserve_existing_nodes(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    name = "test_mai_smart_crop_pack"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    existing = {
        "MAIAutoSeamlessLoop", "mAI_CinematicPost", "MAIBackgroundLightingMatch",
        "MAICompositeLayer", "MAIConditionalLora", "MAIExampleTextNode",
        "MAIFastGPUResize", "MAIFrameLoopFade", "MAIGPUVideoCombine",
        "MAIH3ToLTXFrameAdapter", "MAIImageAspectRatio", "MAIImageGate",
        "MAIImageLogicCheck", "MAIJsonParser", "MAIKrea2ImageConditioning",
        "MAIMaskBoundingBox", "MAIMaskOutline", "MAIPrepareImageForMinimaxH3",
        "MAIRandomLine", "MAISaveTextFile", "MAITextSequenceRandomizer",
        "MAITrimFrameSequence", "MAITypeConverterNode", "MAIVideoLoader",
    }
    assert existing <= package.NODE_CLASS_MAPPINGS.keys()
    for key, display in (("MAIMaskSmartCrop", "mAI Mask Smart Crop"),
                         ("MAIMaskSmartStitch", "mAI Mask Smart Stitch")):
        assert package.NODE_CLASS_MAPPINGS[key].__name__ == key
        assert package.NODE_CLASS_MAPPINGS[key].CATEGORY == "mAI / Mask"
        assert package.NODE_DISPLAY_NAME_MAPPINGS[key] == display


@pytest.mark.parametrize("mask,error", [
    (torch.zeros(1, 3, 3), "spatial dimensions"),
    (torch.zeros(3, 4, 5), "batch"),
    (None, "MASK tensor"),
])
def test_invalid_masks_are_rejected(mask, error):
    with pytest.raises(ValueError, match=error):
        smart_crop(torch.zeros(2, 4, 5, 3), mask, 3, 3)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="mask_threshold"):
        smart_crop(torch.zeros(1, 4, 5, 3), torch.zeros(4, 5), 3, 3, mask_threshold=threshold)
