import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from nodes.image_batch import MAIImageBatch
from nodes.image_gate import MAIImageGate
from utils.image_batch import batch_connected_images


def test_missing_inputs_and_disabled_gates_produce_none():
    assert batch_connected_images([]) is None
    assert batch_connected_images([None, None]) is None
    assert MAIImageBatch().run() == (None,)
    assert MAIImageBatch().run(image_4=MAIImageGate().run(False)[0]) == (None,)


@pytest.mark.parametrize("socket", range(1, 9))
def test_single_available_input_is_unchanged_in_any_socket(socket):
    image = torch.rand(2, 3, 4, 3)
    assert MAIImageBatch().run(**{f"image_{socket}": image})[0] is image


def test_sparse_inputs_keep_socket_order_and_every_frame_without_modification():
    first = torch.arange(72, dtype=torch.float64).reshape(2, 3, 4, 3).transpose(1, 2)
    second = torch.full((1, 4, 3, 3), 99.0, dtype=torch.float64)
    original = first.clone()
    # Keyword insertion order must not affect image socket order.
    result, = MAIImageBatch().run(image_8=second, image_2=first, image_5=None)
    torch.testing.assert_close(result, torch.cat((first, second)))
    torch.testing.assert_close(first, original)
    assert result.dtype == first.dtype and result.device == first.device


def test_gate_toggle_adds_and_removes_only_its_images_without_retaining_old_values():
    batch = MAIImageBatch()
    gate = MAIImageGate()
    first = torch.zeros(1, 2, 3, 3)
    second = torch.ones(2, 2, 3, 3)
    for enabled in (True, False, True):
        result, = batch.run(image_1=first, image_3=gate.run(enabled, second)[0])
        expected = torch.cat((first, second)) if enabled else first
        torch.testing.assert_close(result, expected)
    assert batch.run(image_3=gate.run(True)[0]) == (None,)
    # A fully empty batch can feed another batch just like a disabled gate.
    assert batch.run(image_1=batch.run()[0], image_8=first)[0] is first


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_grayscale_rgb_and_rgba_are_batched_without_channel_changes(channels):
    first = torch.rand(1, 2, 3, channels)
    second = torch.rand(2, 2, 3, channels)
    torch.testing.assert_close(batch_connected_images([first, None, second]), torch.cat((first, second)))


@pytest.mark.parametrize("image,message,error", [
    ("invalid", "image_2.*tensor", TypeError),
    (torch.zeros(2, 3, 3), "image_2.*shape", ValueError),
    (torch.zeros(0, 2, 3, 3), "image_2.*nonempty", ValueError),
    (torch.zeros(1, 2, 3, 3, dtype=torch.uint8), "image_2.*floating", ValueError),
    (torch.zeros(1, 2, 3, 2), "image_2.*RGB", ValueError),
])
def test_invalid_values_have_useful_errors(image, message, error):
    with pytest.raises(error, match=message):
        batch_connected_images([None, image])


@pytest.mark.parametrize("image,message", [
    (torch.zeros(1, 2, 3, 4), "channels"),
    (torch.zeros(1, 2, 3, 3, dtype=torch.float64), "dtype and device"),
    (torch.empty(1, 2, 3, 3, device="meta"), "dtype and device"),
])
def test_incompatible_inputs_fail_clearly(image, message):
    with pytest.raises(ValueError, match=message):
        batch_connected_images([torch.zeros(1, 2, 3, 3), None, image])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_resize_matches_first_available_image_and_preserves_every_frame(dtype):
    first = torch.full((1, 3, 3, 3), 0.25, dtype=dtype)
    pattern = torch.tensor([[0.0, 0.25], [0.5, 0.75]], dtype=dtype)
    second = pattern.reshape(1, 2, 2, 1).expand(2, 2, 2, 3)
    original = second.clone()
    result, = MAIImageBatch().run(image_2=first, image_6=second)
    expected = torch.tensor([
        [0.0, 0.125, 0.25],
        [0.25, 0.375, 0.5],
        [0.5, 0.625, 0.75],
    ], dtype=dtype).reshape(1, 3, 3, 1).expand(2, 3, 3, 3)
    assert result.shape == (3, 3, 3, 3)
    assert result.dtype == dtype and result.device == first.device
    torch.testing.assert_close(result[:1], first)
    torch.testing.assert_close(result[1:], expected)
    torch.testing.assert_close(second, original)


def test_downsize_stretches_to_reference_without_cropping():
    first = torch.zeros(1, 2, 3, 1)
    second = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6, 1) / 24
    result, = MAIImageBatch().run(image_1=first, image_8=second)
    expected = torch.tensor([[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]]) / 24
    assert result.shape == (2, 2, 3, 1)
    torch.testing.assert_close(result[1, ..., 0], expected)


def test_gate_toggle_changes_reference_to_first_remaining_image():
    gate = MAIImageGate()
    first = torch.zeros(1, 3, 5, 3)
    second = torch.ones(2, 7, 2, 3)
    batch = MAIImageBatch()
    for enabled in (True, False, True):
        result, = batch.run(image_1=gate.run(enabled, first)[0], image_4=second)
        if enabled:
            assert result.shape == (3, 3, 5, 3)
            torch.testing.assert_close(result[:1], first)
            torch.testing.assert_close(result[1:], torch.ones(2, 3, 5, 3))
        else:
            assert result is second


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_resize_on_cuda_keeps_images_on_their_device():
    first = torch.zeros(1, 3, 5, 4, device="cuda", dtype=torch.float16)
    second = torch.ones(2, 7, 2, 4, device="cuda", dtype=torch.float16)
    result, = MAIImageBatch().run(image_1=first, image_2=second)
    assert result.shape == (3, 3, 5, 4)
    assert result.device == first.device and result.dtype == first.dtype
    torch.testing.assert_close(result[1:], torch.ones_like(result[1:]))


def test_node_contract_and_pack_registration_preserve_existing_nodes(monkeypatch):
    assert MAIImageBatch.CATEGORY == "mAI / Image"
    assert MAIImageBatch.RETURN_TYPES == ("IMAGE",)
    assert MAIImageBatch.RETURN_NAMES == ("image",)
    inputs = MAIImageBatch.INPUT_TYPES()
    assert inputs["required"] == {}
    assert list(inputs["optional"]) == [f"image_{number}" for number in range(1, 9)]
    assert all(spec[0] == "IMAGE" for spec in inputs["optional"].values())
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "test_mai_image_batch_pack", root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS["MAIImageBatch"].__name__ == "MAIImageBatch"
    assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIImageBatch"] == "mAI image batch"
    assert "MAIImageGate" in package.NODE_CLASS_MAPPINGS
    assert "MAIFastGPUResize" in package.NODE_CLASS_MAPPINGS
