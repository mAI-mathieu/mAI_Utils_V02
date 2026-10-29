import pytest
import torch

from nodes.fast_gpu_resize import MAIFastGPUResize
from utils.resize_kernels import RESIZE_METHODS, RESIZE_MODES
from utils.resize_masks import prepare_resize_mask


@pytest.fixture(autouse=True, scope="module")
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("mode", RESIZE_MODES)
@pytest.mark.parametrize("method", RESIZE_METHODS)
@pytest.mark.parametrize("chunk_size", [0, 2])
def test_mask_stays_aligned_with_image_for_all_modes_and_methods(mode, method, chunk_size):
    mask = torch.rand(3, 9, 13)
    image = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
    before = mask.clone()
    result, w, h, _, output_mask, _ = MAIFastGPUResize().run(
        image, 8, 6, resize_mode=mode, method=method,
        multiple_of=4, chunk_size=chunk_size, mask=mask,
    )
    assert output_mask.shape == (3, h, w)
    torch.testing.assert_close(output_mask, result[..., 0])
    assert torch.equal(mask, before)


@pytest.mark.parametrize("mode", RESIZE_MODES)
def test_missing_mask_returns_zero_mask_at_actual_output_size(mode):
    image = torch.rand(3, 9, 13, 3)
    result, w, h, _, mask, _ = MAIFastGPUResize().run(
        image, 8, 6, resize_mode=mode, multiple_of=4,
    )
    assert mask.shape == (3, h, w)
    assert mask.dtype == result.dtype and mask.device == result.device
    assert not mask.any()


@pytest.mark.parametrize("shape", [(9, 13), (1, 9, 13)])
def test_single_mask_broadcasts_across_image_batch(shape):
    mask = torch.arange(117).float().div(116).reshape(shape)
    image = torch.rand(3, 9, 13, 3)
    output_mask = MAIFastGPUResize().run(image, 8, 6, resize_mode="center_crop", mask=mask)[4]
    expected = mask.reshape(1, 9, 13)[:, 1:7, 2:10].expand(3, -1, -1)
    assert torch.equal(output_mask, expected)


@pytest.mark.parametrize("mode,expected", [
    ("pad_to_multiple", [[0, 0, 0, 0], [1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]]),
    ("pad_to_multiple_edge", [[1, 2, 3, 3], [1, 2, 3, 3], [4, 5, 6, 6], [4, 5, 6, 6]]),
])
def test_mask_padding_has_expected_borders_and_preserves_values(mode, expected):
    mask = torch.arange(1, 7).float().reshape(1, 2, 3)
    image = torch.rand(1, 2, 3, 3)
    output_mask = MAIFastGPUResize().run(
        image, resize_mode=mode, multiple_of=4, mask=mask,
    )[4]
    assert torch.equal(output_mask, torch.tensor([expected], dtype=torch.float32))


@pytest.mark.parametrize("mask,message", [
    ("invalid", "non-empty torch tensor"),
    (torch.zeros(9), "non-empty torch tensor"),
    (torch.zeros(1, 9, 13, 1), "non-empty torch tensor"),
    (torch.zeros(0, 9, 13), "non-empty torch tensor"),
    (torch.zeros(1, 9, 13, dtype=torch.uint8), "must use float32"),
    (torch.zeros(1, 8, 13), "must match the source image"),
    (torch.zeros(1, 9, 12), "must match the source image"),
    (torch.zeros(2, 9, 13), "match the image batch size"),
])
def test_invalid_masks_fail_with_clear_errors_before_image_processing(mask, message, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid masks must be rejected before resizing")

    monkeypatch.setattr("nodes.fast_gpu_resize.resize_image_batch", forbidden)
    with pytest.raises(ValueError, match=message):
        MAIFastGPUResize().run(torch.zeros(3, 9, 13, 3), mask=mask)


def test_preparation_rejects_invalid_source_shape():
    with pytest.raises(ValueError, match="image must be a non-empty torch tensor"):
        prepare_resize_mask(torch.zeros(1, 9, 13), (9, 13, 3))


@pytest.mark.parametrize("precision,dtype", [
    ("auto", torch.float64), ("fp32", torch.float32),
    ("fp16", torch.float16), ("bf16", torch.bfloat16),
])
def test_mask_precision_is_honored(precision, dtype):
    mask = torch.rand(2, 9, 13, dtype=torch.float64)
    image = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
    result, _, _, _, output_mask, _ = MAIFastGPUResize().run(
        image, 8, 6, resize_mode="exact", method="nearest", precision=precision, mask=mask,
    )
    assert output_mask.dtype == dtype and result.dtype == dtype
    assert torch.equal(output_mask, result[..., 0])


@pytest.mark.parametrize("mode", ["pad_to_multiple", "pad_to_multiple_edge", "center_crop"])
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("chunk_size", [0, 2])
def test_padding_and_crop_preserve_image_and_mask_despite_precision(mode, precision, chunk_size, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pixel-preserving modes must not resample")

    monkeypatch.setattr("utils.resize_kernels.resize_native", forbidden)
    monkeypatch.setattr("utils.resize_kernels.build_sampling_weights", forbidden)
    # Noncontiguous values outside [0,1] and extra precision catch casts/clamps.
    image = torch.linspace(-0.123456789, 1.987654321, 3 * 18 * 26 * 4,
                           dtype=torch.float64).reshape(3, 18, 26, 4)[:, ::2, ::2]
    mask = image[..., 0].float()
    before_image, before_mask = image.clone(), mask.clone()
    result, w, h, used, output_mask, _ = MAIFastGPUResize().run(
        image, 8, 6, resize_mode=mode, method="lanczos4", antialias=True,
        multiple_of=8, precision=precision, chunk_size=chunk_size, mask=mask,
    )
    assert used == "identity"
    assert result.dtype == image.dtype and output_mask.dtype == mask.dtype
    if mode == "center_crop":
        assert (w, h) == (8, 6)
        assert torch.equal(result, image[:, 1:7, 2:10])
        assert torch.equal(output_mask, mask[:, 1:7, 2:10])
    else:
        assert (w, h) == (16, 16)
        assert torch.equal(result[:, 3:12, 1:14], image)
        assert torch.equal(output_mask[:, 3:12, 1:14], mask)
    assert torch.equal(image, before_image) and torch.equal(mask, before_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("image_device,mask_device,device", [
    ("cpu", "cpu", "gpu"), ("cuda", "cpu", "auto"),
    ("cpu", "cuda", "auto"), ("cuda", "cuda", "cpu"),
])
def test_mask_follows_image_processing_device(image_device, mask_device, device):
    source = torch.rand(2, 9, 13)
    image = source.unsqueeze(-1).expand(-1, -1, -1, 3).to(image_device)
    result, _, _, _, mask, _ = MAIFastGPUResize().run(
        image, 8, 6, method="nearest", device=device, mask=source.to(mask_device),
    )
    assert mask.device == result.device
    torch.testing.assert_close(mask, result[..., 0])
