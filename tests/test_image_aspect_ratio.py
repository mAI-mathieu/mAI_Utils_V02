from types import SimpleNamespace

import pytest

from nodes.image_aspect_ratio import MAIImageAspectRatio


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1, 1080, 1920, 3), "16:9"),
        ((4, 1920, 1080, 4), "9/16"),
        ((2, 1024, 1024, 3), "1:1"),
    ],
)
def test_reads_image_dimensions_and_returns_one_string(shape, expected):
    assert MAIImageAspectRatio().run(SimpleNamespace(shape=shape)) == (expected,)


@pytest.mark.parametrize("shape", [None, (100, 100, 3), (0, 100, 100, 3), (1, 0, 100, 3)])
def test_rejects_missing_or_empty_images(shape):
    with pytest.raises(ValueError, match="non-empty shape"):
        MAIImageAspectRatio().run(SimpleNamespace(shape=shape))
