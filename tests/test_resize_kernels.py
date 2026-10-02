import math

import pytest
import torch
import torch.nn.functional as F

from utils.resize_kernels import (
    CUSTOM_METHODS, NATIVE_METHODS, RESIZE_METHODS, apply_multiple_of,
    bc_cubic_kernel, build_sampling_weights, calculate_target_size,
    choose_chunk_size, lanczos_kernel, resize_image_batch, select_auto_resize_method,
)


@pytest.fixture(autouse=True, scope="module")
def small_tensor_threads():
    # Small synthetic tests should not launch dozens of CPU worker threads.
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("method", RESIZE_METHODS)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_dimensions_constant_color_and_channels(method, channels):
    image = torch.full((2, 11, 17, channels), 0.37)
    result, w, h, used = resize_image_batch(image, 9, 7, method=method)
    assert result.shape == (2, 7, 9, channels)
    assert (w, h) == (9, 7)
    assert used == ("lanczos3" if method == "auto" else method)
    torch.testing.assert_close(result, torch.full_like(result, 0.37))
    assert result.dtype == image.dtype and result.device == image.device


@pytest.mark.parametrize("method", RESIZE_METHODS)
@pytest.mark.parametrize("chunks", [0, 32])
def test_121_frame_batch_preservation_and_order(method, chunks):
    levels = torch.linspace(0, 1, 121)
    image = levels[:, None, None, None].expand(121, 9, 13, 3)
    result, _, _, _ = resize_image_batch(image, 7, 5, method=method, chunk_size=chunks)
    assert result.shape == (121, 5, 7, 3)
    torch.testing.assert_close(result.mean((1, 2, 3)), levels, atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(result[0]) == 0
    torch.testing.assert_close(result[-1], torch.ones_like(result[-1]))


@pytest.mark.parametrize("method", RESIZE_METHODS)
def test_identity_returns_original_without_clamping(method):
    image = torch.randn(3, 5, 7, 4)
    result, w, h, used = resize_image_batch(image, 7, 5, method=method)
    assert result is image
    assert (w, h, used) == (7, 5, "identity")


@pytest.mark.parametrize("method", NATIVE_METHODS)
@pytest.mark.parametrize("antialias", [False, True])
def test_native_matches_direct_interpolate(method, antialias):
    image = torch.rand(3, 8, 12, 3)
    kwargs = {"align_corners": False, "antialias": antialias} if method in ("bilinear", "bicubic") else {}
    expected = F.interpolate(image.permute(0, 3, 1, 2), size=(5, 9), mode=method, **kwargs)
    if method == "bicubic":
        expected.clamp_(0, 1)
    actual = resize_image_batch(image, 9, 5, method=method, antialias=antialias)[0]
    torch.testing.assert_close(actual, expected.permute(0, 2, 3, 1))


def test_fit_padding_and_fill_center_crop():
    image = torch.ones(1, 4, 8, 3)
    fit = resize_image_batch(image, 8, 8, "keep_aspect_fit", "nearest")[0]
    assert fit.shape == (1, 8, 8, 3)
    assert not fit[:, :2].any() and not fit[:, 6:].any()
    assert fit[:, 2:6].eq(1).all()
    columns = torch.arange(8).float().div(7).view(1, 1, 8, 1).expand(2, 4, 8, 1)
    fill = resize_image_batch(columns, 4, 4, "keep_aspect_fill", "nearest")[0]
    torch.testing.assert_close(fill, columns[:, :, 2:6])


def test_scaled_fill_and_odd_padding():
    image = torch.rand(2, 3, 8, 4)
    output = resize_image_batch(image, 5, 7, "keep_aspect_fill", "nearest")[0]
    geometry = calculate_target_size(8, 3, 5, 7, "keep_aspect_fill")
    assert (geometry.resize_width, geometry.resize_height) == (19, 7)
    expected = F.interpolate(image.permute(0, 3, 1, 2), (7, 19), mode="nearest")[:, :, :, 7:12]
    torch.testing.assert_close(output, expected.permute(0, 2, 3, 1))
    fit = resize_image_batch(torch.ones(1, 2, 4, 1), 4, 5, "keep_aspect_fit", "nearest")[0]
    assert fit[:, 0].eq(0).all() and fit[:, 1:3].eq(1).all() and fit[:, 3:].eq(0).all()


@pytest.mark.parametrize("source,target,expected", [
    ((1920, 1080), (1024, 1024), (1024, 576)),
    ((1080, 1920), (1024, 1024), (576, 1024)),
    ((800, 800), (640, 480), (480, 480)),
    ((16, 9), (64, 36), (64, 36)),
    ((3, 2), (5, 5), (5, 3)),
    ((1000, 1), (10, 10), (10, 1)),
    ((1, 1000), (10, 10), (1, 10)),
])
def test_keep_aspect_dimensions(source, target, expected):
    geometry = calculate_target_size(*source, *target, "keep_aspect")
    assert (geometry.width, geometry.height) == expected
    assert (geometry.resize_width, geometry.resize_height) == expected


def test_keep_aspect_rounds_output_to_multiple():
    geometry = calculate_target_size(1920, 1080, 1000, 1000, "keep_aspect", 8)
    assert (geometry.width, geometry.height) == (1000, 560)
    assert (geometry.resize_width, geometry.resize_height) == (1000, 560)


@pytest.mark.parametrize("method", RESIZE_METHODS)
@pytest.mark.parametrize("chunk_size", [0, 2])
def test_keep_aspect_preserves_entire_image_and_batch(method, chunk_size):
    image = torch.linspace(0.1, 0.9, 3 * 6 * 10 * 4).reshape(3, 6, 10, 4)
    before = image.clone()
    result, w, h, used = resize_image_batch(
        image, 5, 5, "keep_aspect", method, chunk_size=chunk_size,
    )
    expected = resize_image_batch(image, 5, 3, "exact", method)[0]
    assert result.shape == (3, 3, 5, 4) and (w, h) == (5, 3)
    assert used == ("lanczos3" if method == "auto" else method)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(image, before)


def test_keep_aspect_identity_uses_fitted_dimensions():
    image = torch.ones(2, 9, 16, 3)
    result, w, h, used = resize_image_batch(image, 16, 16, "keep_aspect")
    assert result is image and (w, h, used) == (16, 9, "identity")


@pytest.mark.parametrize("dimension,multiple,expected", [(1023,8,1024),(1019,8,1016),(1020,8,1024),(1018,8,1016),(1,64,64),(1,1,1)])
def test_multiple_rounding(dimension, multiple, expected):
    assert apply_multiple_of(dimension, multiple) == expected


@pytest.mark.parametrize("mode", ["exact", "keep_aspect_fit", "keep_aspect_fill"])
def test_canvas_multiple_of(mode):
    result, w, h, _ = resize_image_batch(torch.ones(1, 3, 7, 1), 11, 13, mode, multiple_of=8)
    assert (w, h) == (8, 16)
    assert result.shape == (1, h, w, 1)


@pytest.mark.parametrize("target,expected", [((20,10),"identity"),((13,10),"lanczos3"),((14,8),"lanczos2"),((30,15),"bicubic"),((31,16),"catmull_rom"),((10,20),"lanczos3")])
def test_auto_thresholds_and_mixed_axis_resize(target, expected):
    assert select_auto_resize_method(20, 10, *target) == expected


@pytest.mark.parametrize("method", CUSTOM_METHODS)
def test_edges_do_not_wrap(method):
    image = torch.zeros(1, 8, 32, 1)
    image[:, :, 0] = 1
    result = resize_image_batch(image, 47, 11, method=method)[0]
    assert result[:, :, -8:].abs().max() == 0
    assert result[:, :, 0].mean() > 0.5


@pytest.mark.parametrize("method", CUSTOM_METHODS)
def test_antialias_reduces_high_frequency_aliases(method):
    image = (torch.arange(127) % 2).float().view(1, 1, 127, 1).expand(1, 8, 127, 1)
    smooth = resize_image_batch(image, 37, 8, method=method, antialias=True)[0][:, :, 3:-3]
    sharp = resize_image_batch(image, 37, 8, method=method, antialias=False)[0][:, :, 3:-3]
    assert (smooth - 0.5).square().mean() < (sharp - 0.5).square().mean() * 0.05


def _reference_kernel(x, method):
    """Independent scalar definition, used only on tiny reference images."""
    x = abs(x)
    if method.startswith("lanczos"):
        a = int(method[-1])
        if x >= a:
            return 0.0
        sinc = lambda v: 1.0 if v == 0 else math.sin(math.pi*v) / (math.pi*v)
        return sinc(x) * sinc(x/a)
    b, c = (1/3, 1/3) if method == "mitchell" else (0, 0.5)
    if x < 1:
        return ((12-9*b-6*c)*x**3 + (-18+12*b+6*c)*x**2 + 6-2*b)/6
    if x < 2:
        return ((-b-6*c)*x**3 + (6*b+30*c)*x**2 + (-12*b-48*c)*x+8*b+24*c)/6
    return 0.0


@pytest.mark.parametrize("method", CUSTOM_METHODS)
@pytest.mark.parametrize("src,dst,aa", [(9,4,True),(5,8,True),(9,4,False)])
def test_custom_against_independent_scalar_reference(method, src, dst, aa):
    values = [0.12 + 0.07*i for i in range(src)]
    radius = int(method[-1]) if method.startswith("lanczos") else 2
    stretch = max(1, src/dst) if aa else 1
    expected = []
    for out in range(dst):
        center = (out+0.5)*src/dst-0.5
        support = radius*stretch
        indices = range(math.floor(center-support), math.ceil(center+support)+1)
        weights = [_reference_kernel((center-i)/stretch, method) for i in indices]
        value = sum(values[min(src-1,max(0,i))]*w for i,w in zip(indices,weights))/sum(weights)
        expected.append(min(1,max(0,value)))
    image = torch.tensor(values, dtype=torch.float64).view(1, 1, src, 1)
    actual = resize_image_batch(image, dst, 1, method=method, antialias=aa)[0].flatten()
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_kernel_definitions_and_support():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    assert lanczos_kernel(x, 3)[0] == 1
    assert lanczos_kernel(x, 3)[3:].eq(0).all()
    torch.testing.assert_close(bc_cubic_kernel(x, 1/3, 1/3)[:2], torch.tensor([8/9, 1/18]))
    assert bc_cubic_kernel(x, 0, 0.5)[0] == 1
    assert bc_cubic_kernel(x, 0, 0.5)[2:].eq(0).all()
    indices, weights = build_sampling_weights(128, 16, "lanczos3", True, "cpu")
    assert indices.shape[0] >= 48
    assert indices.min() == 0 and indices.max() == 127
    torch.testing.assert_close(weights.sum(0), torch.ones(16))


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("method", NATIVE_METHODS + CUSTOM_METHODS)
@pytest.mark.parametrize("antialias", [False, True])
def test_cpu_cuda_dtypes_and_noncontiguous_inputs(device, dtype, method, antialias):
    image = torch.rand(2, 13, 19, 4, device=device, dtype=dtype)[:, ::2, ::2]
    before = image.clone()
    result = resize_image_batch(image, 6, 11, method=method, antialias=antialias, chunk_size=1)[0]
    assert result.shape == (2, 11, 6, 4)
    assert result.dtype == dtype and result.device == image.device
    assert torch.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1
    torch.testing.assert_close(image, before)


@pytest.mark.parametrize("shape,target", [((1,1,1,1),(2,3)),((1,1,7,3),(1,1)),((1,7,1,4),(9,1)),((1,2,31,1),(1,17))])
@pytest.mark.parametrize("method", CUSTOM_METHODS)
def test_tiny_images_and_unusual_aspect_ratios(shape, target, method):
    result = resize_image_batch(torch.full(shape, 0.4), *target, method=method)[0]
    torch.testing.assert_close(result, torch.full_like(result, 0.4))


@pytest.mark.parametrize("method", CUSTOM_METHODS)
def test_chunked_matches_unchunked_and_does_not_modify_input(method):
    image = torch.rand(7, 13, 23, 3)
    before = image.clone()
    full = resize_image_batch(image, 31, 8, method=method)[0]
    chunked = resize_image_batch(image, 31, 8, method=method, chunk_size=3)[0]
    torch.testing.assert_close(full, chunked)
    torch.testing.assert_close(image, before)


@pytest.mark.parametrize("method", CUSTOM_METHODS)
def test_vertical_filter_matches_horizontal_under_transpose(method):
    image = torch.rand(2, 13, 23, 4)
    normal = resize_image_batch(image, 31, 8, method=method)[0]
    transposed = resize_image_batch(image.transpose(1, 2), 8, 31, method=method)[0]
    torch.testing.assert_close(normal, transposed.transpose(1, 2), atol=2e-6, rtol=2e-6)


def test_auto_chunk_budget_and_explicit_override():
    auto = choose_chunk_size(121, 3, (1080,1920), (576,1024), (576,1024), 0, True)
    assert 1 < auto < 121
    assert choose_chunk_size(121,3,(1080,1920),(576,1024),(576,1024),32,True) == 32
    assert choose_chunk_size(121,3,(1080,1920),(576,1024),(576,1024),0,False) == 121


@pytest.mark.parametrize("kwargs", [{"width":0},{"height":-1},{"multiple_of":0},{"width":True},{"chunk_size":-1},{"chunk_size":1.5},{"method":"fake"},{"resize_mode":"fake"},{"precision":"fake"},{"antialias":"true"}])
def test_invalid_settings(kwargs):
    settings = {"width":4,"height":4, **kwargs}
    with pytest.raises(ValueError):
        resize_image_batch(torch.zeros(1,4,4,3), **settings)


@pytest.mark.parametrize("image", [None, torch.zeros(4,4,3), torch.zeros(0,4,4,3), torch.zeros(1,4,0,3), torch.zeros(1,4,4,0), torch.zeros(1,4,4,3,dtype=torch.uint8)])
def test_invalid_images(image):
    with pytest.raises(ValueError):
        resize_image_batch(image,4,4)
