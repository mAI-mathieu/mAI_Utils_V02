import pytest

from nodes.image_gate import MAIImageGate


class ImageSentinel:
    """Image tensors must not be truth-tested, copied, or processed by the gate."""

    def __bool__(self):
        raise AssertionError("Image truth value must not be evaluated")


def test_toggle_preserves_image_identity_and_does_not_retain_disabled_image():
    node = MAIImageGate()
    image = ImageSentinel()
    assert node.run(True, image)[0] is image
    assert node.run(False, image) == (None,)
    assert node.run(True, image)[0] is image


@pytest.mark.parametrize("enabled", [True, False])
def test_missing_optional_image_is_safe(enabled):
    assert MAIImageGate().run(enabled) == (None,)
    assert MAIImageGate().check_lazy_status(enabled) == []


def test_lazy_image_is_requested_only_when_enabled_and_not_available():
    node = MAIImageGate()
    assert node.INPUT_TYPES()["optional"]["image"][1]["lazy"] is True
    assert node.check_lazy_status(False) == []
    assert node.check_lazy_status(False, None) == []
    assert node.check_lazy_status(True, None) == ["image"]
    assert node.check_lazy_status(True, ImageSentinel()) == []
    assert node.check_lazy_status(False, ImageSentinel()) == []
