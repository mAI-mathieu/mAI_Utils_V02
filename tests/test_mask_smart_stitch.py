import pytest
import torch

from nodes.mask_smart_stitch import MAIMaskSmartStitch
from utils.mask_smart_crop import CONTEXT_TYPE, smart_crop
from utils.mask_smart_stitch import smart_stitch


@pytest.mark.parametrize("edge_mode", ["shift", "pad"])
@pytest.mark.parametrize("mode", ["full_crop", "mask_only", "mask_feather"])
@pytest.mark.parametrize("target", [(7, 5), (25, 23)])
def test_native_roundtrip_is_pixel_identical_including_padding_soft_masks_and_batches(edge_mode, mode, target):
    image = torch.rand(2, 15, 19, 3)
    mask = torch.zeros(2, 15, 19)
    mask[0, 0:3, 0:2] = 0.3
    mask[1, 12:15, 16:19] = 0.9
    image_before, mask_before = image.clone(), mask.clone()
    cropped = smart_crop(image, mask, *target, edge_mode=edge_mode)
    stitched = smart_stitch(image, cropped[0], cropped[3], feather=2, composite_mode=mode)
    assert torch.equal(stitched, image)
    assert torch.equal(image, image_before)
    assert torch.equal(mask, mask_before)


@pytest.mark.parametrize("edge_mode", ["shift", "pad"])
@pytest.mark.parametrize("upscale", [False, True])
def test_full_crop_changes_exactly_the_full_crop_mask_for_every_transform(edge_mode, upscale):
    image = torch.zeros(2, 21, 29, 3)
    mask = torch.zeros(2, 21, 29)
    mask[0, 0:3, 0:3] = 1
    mask[1, 9:21, 16:29] = 1
    for target in ((7, 7), (14, 10), (35, 35)):
        crop = smart_crop(image, mask, *target, edge_mode=edge_mode, allow_upscale=upscale)
        result = smart_stitch(image, torch.ones_like(crop[0]), crop[3], composite_mode="full_crop")
        assert result.shape == image.shape
        assert torch.allclose(result, crop[2][..., None].expand_as(result), atol=1e-6)


def test_default_mask_stays_exact_in_source_space_after_downscale():
    image = torch.zeros(1, 40, 50, 3)
    mask = torch.zeros(1, 40, 50)
    mask[:, 2:28, 3:34] = 0.6
    mask[:, 10:20, 15:25] = 0
    crop = smart_crop(image, mask, 12, 12)
    result = smart_stitch(image, torch.ones_like(crop[0]), crop[3])
    assert torch.equal(result[mask == 0], image[mask == 0])
    assert torch.allclose(result, mask[..., None].expand_as(result), atol=1e-6)


def test_explicit_crop_mask_maps_through_padding_and_scaling_and_broadcasts():
    image = torch.zeros(2, 20, 24, 3)
    source_mask = torch.ones(1, 20, 24)
    crop = smart_crop(image, source_mask, 8, 8, edge_mode="pad")
    mask = torch.full((1, 8, 8), 0.25)
    result = smart_stitch(image, torch.ones_like(crop[0]), crop[3], mask=mask)
    assert torch.allclose(result, torch.full_like(image, 0.25), atol=1e-6)


def test_feather_softens_edges_but_preserves_pixels_outside_crop():
    image = torch.zeros(1, 30, 30, 3)
    mask = torch.zeros(1, 30, 30)
    mask[:, 12:18, 12:18] = 1
    crop = smart_crop(image, mask, 16, 16)
    generated = torch.ones_like(crop[0])
    hard = smart_stitch(image, generated, crop[3], feather=3, composite_mode="mask_only")
    soft = smart_stitch(image, generated, crop[3], feather=3)
    assert torch.equal(hard, mask[..., None].expand_as(image))
    assert 0 < soft[0, 11, 15, 0] < 1
    assert 0 < soft[0, 12, 15, 0] < 1
    assert torch.equal(soft[crop[2] == 0], image[crop[2] == 0])


def test_empty_semantic_mask_and_context_without_mask_have_safe_fallbacks():
    image = torch.zeros(1, 10, 10, 3)
    crop = smart_crop(image, torch.zeros(10, 10), 6, 6)
    generated = torch.ones_like(crop[0])
    assert torch.equal(smart_stitch(image, generated, crop[3]), image)
    del crop[3]["items"][0]["source_mask"]
    fallback = smart_stitch(image, generated, crop[3])
    assert torch.equal(fallback, crop[2][..., None].expand_as(image))


def test_resized_roundtrip_for_smooth_image_is_close_and_outside_mask_is_exact():
    ramp = torch.linspace(0, 1, 80)[None, None, :, None].expand(1, 60, 80, 3)
    mask = torch.zeros(1, 60, 80)
    mask[:, 10:50, 10:70] = 1
    crop = smart_crop(ramp, mask, 32, 32)
    result = smart_stitch(ramp, crop[0], crop[3])
    assert torch.equal(result[mask == 0], ramp[mask == 0])
    assert torch.allclose(result, ramp, atol=0.005)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_device_and_dtype_are_preserved(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    image = torch.rand(1, 17, 19, 3, device=device, dtype=dtype)
    mask = torch.zeros(17, 19, device=device, dtype=dtype)
    mask[1:16, 1:18] = 0.5
    for target in ((23, 23), (8, 8)):
        crop = smart_crop(image, mask, *target, edge_mode="pad")
        result = smart_stitch(image, crop[0], crop[3], feather=2)
        assert result.dtype == image.dtype and result.device == image.device
        assert all(t.dtype == dtype and t.device == image.device for t in crop[:3])
        if crop[-1] == 1:
            assert torch.equal(result, image)


def test_node_contract_and_no_optional_inputs():
    node = MAIMaskSmartStitch()
    assert node.RETURN_TYPES == ("IMAGE",)
    assert node.RETURN_NAMES == ("image",)
    assert node.CATEGORY == "mAI / Mask"
    assert node.INPUT_TYPES()["required"]["crop_context"] == (CONTEXT_TYPE,)
    assert node.INPUT_TYPES()["optional"]["composite_mode"][1]["default"] == "mask_feather"
    original = torch.rand(1, 4, 4, 3)
    crop = smart_crop(original, torch.ones(4, 4), 4, 4)
    assert torch.equal(node.run(original, crop[0], crop[3])[0], original)


@pytest.mark.parametrize("change,error", [
    ("context", "must come from"),
    ("version", "version"),
    ("batch", "batches must match"),
    ("generated_size", "dimensions"),
    ("original_size", "dimensions"),
    ("channels", "channels"),
    ("mask", "spatial dimensions"),
    ("mask_batch", "batch"),
    ("mode", "composite_mode"),
    ("feather", "feather"),
])
def test_incompatible_inputs_are_rejected(change, error):
    original = torch.rand(2, 10, 12, 3)
    crop = smart_crop(original, torch.ones(10, 12), 6, 6)
    kwargs = dict(original_image=original, generated_image=crop[0], crop_context=crop[3])
    if change == "context":
        kwargs["crop_context"] = {}
    elif change == "version":
        crop[3]["version"] = -1
    elif change == "batch":
        kwargs["generated_image"] = crop[0][:1]
    elif change == "generated_size":
        kwargs["generated_image"] = torch.zeros(2, 7, 6, 3)
    elif change == "original_size":
        kwargs["original_image"] = torch.zeros(2, 11, 12, 3)
    elif change == "channels":
        kwargs["generated_image"] = crop[0][..., :1]
    elif change == "mask":
        kwargs["mask"] = torch.zeros(10, 12)
    elif change == "mask_batch":
        kwargs["mask"] = torch.zeros(3, 6, 6)
    elif change == "mode":
        kwargs["composite_mode"] = "unknown"
    else:
        kwargs["feather"] = -1
    with pytest.raises(ValueError, match=error):
        smart_stitch(**kwargs)
