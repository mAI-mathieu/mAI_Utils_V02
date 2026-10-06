"""Independent synthetic ground truths; no ComfyUI runtime or video codecs."""

from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch

from nodes.auto_seamless_loop import MAIAutoSeamlessLoop
from utils import seamless_loop as loop


def options(**kwargs):
    return replace(loop.LoopOptions(device="cpu", quality="fast", max_trim_start=2,
                                    max_trim_end=2, max_fade=4), **kwargs)


def periodic_clip(n=16):
    # Eight-pixel periodic texture translates one pixel per frame and wraps.
    stripe = (torch.arange(24) % 8 < 3).float()
    base = stripe[None, :, None].expand(16, 24, 3) * 0.7 + 0.1
    return torch.stack([base.roll(i, dims=1) for i in range(n)])


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("space", ["srgb", "linear"])
def test_exact_overlap_construction_and_both_connections(k, space):
    source = torch.arange(12).float()[:, None, None, None].expand(12, 3, 5, 3) / 12
    original = source.clone()
    t = source[1:11]
    result = loop.render_cycle(source, loop.LoopCandidate(1, 1, k), space, chunk=2)
    assert len(result) == len(t) - k
    if not k:
        torch.testing.assert_close(result, t)
    else:
        torch.testing.assert_close(result[:len(t) - 2 * k], t[k:len(t) - k])
        if k >= 2:
            # Inclusive endpoints continue original motion into/out of the bridge.
            torch.testing.assert_close(result[len(t) - 2 * k], t[-k])
            torch.testing.assert_close(result[-1], t[k - 1])
            torch.testing.assert_close(result[0], t[k])
            torch.testing.assert_close(result[len(t) - 2 * k - 1], t[-k - 1])
        elif space == "srgb":
            torch.testing.assert_close(result[-1], (t[0] + t[-1]) / 2)
    torch.testing.assert_close(source, original)
    assert result.min() >= 0 and result.max() <= 1
    assert result.data_ptr() != source.data_ptr()


def test_weight_endpoints_and_linear_light_rgba():
    assert loop.blend_weights(1).tolist() == [0.5]
    assert loop.blend_weights(3).tolist() == [0.0, 0.5, 1.0]
    black, white = torch.zeros(1, 1, 1, 4), torch.ones(1, 1, 1, 4)
    linear = loop.blend_frames(black, white, torch.tensor(0.5), "linear")
    torch.testing.assert_close(linear[..., :3], torch.full((1, 1, 1, 3), 0.73535698))
    assert linear[..., 3].item() == 0.5


@pytest.mark.parametrize("n", [1, 2, 3, 16])
def test_static_and_short_clips_are_unchanged(n):
    x = torch.full((n, 8, 12, 4), 0.3)
    result, report = loop.optimize_loop(x, options())
    assert report["selected"] == {"trim_start": 0, "trim_end": 0, "overlap": 0}
    assert report["selection"] == "unchanged_satisfactory"
    torch.testing.assert_close(result, x)
    assert report["output_duration_seconds"] == n / 24
    json.dumps(report, allow_nan=False)


def test_periodic_translation_keeps_every_frame():
    x = periodic_clip()
    result, report = loop.optimize_loop(x, options())
    torch.testing.assert_close(result, x)
    assert report["selected"]["overlap"] == 0
    # A valid periodic loop does not need matching first/last images.
    assert not torch.equal(x[0], x[-1])


def test_duplicate_endpoint_is_not_automatically_deleted():
    base = periodic_clip()
    x = torch.cat((base, base[:1]))
    result, report = loop.optimize_loop(x, options(max_trim_start=0, max_trim_end=0, max_fade=0))
    torch.testing.assert_close(result, x)
    assert len(result) == 17


def test_bad_exposure_frame_can_be_trimmed_without_fading():
    x = torch.cat((torch.ones(1, 16, 24, 3), torch.full((15, 16, 24, 3), 0.2)))
    result, report = loop.optimize_loop(x, options(max_trim_start=1, max_trim_end=0, max_fade=4))
    assert report["selected"] == {"trim_start": 1, "trim_end": 0, "overlap": 0}
    assert result.shape[0] == 15
    torch.testing.assert_close(result, torch.full_like(result, 0.2))


def score_candidates(x, candidates, opts):
    return loop._score_candidates(x, candidates, len(x), opts, 4, None)


def test_exposure_discontinuity_is_visible_in_color_metric():
    x = torch.linspace(0.1, 0.9, 16)[:, None, None, None].expand(16, 12, 18, 3)
    _, metrics = score_candidates(x, [loop.LoopCandidate()], options())
    assert metrics[0, loop.METRICS.index("exposure")] > 0.5


def contrast_texture():
    checker = ((torch.arange(12)[None, :] + torch.arange(8)[:, None]) % 2).float()
    return (checker * 0.24 + 0.38)[..., None].expand(8, 12, 3)


def test_equal_mean_contrast_jump_is_reported_and_fading_reduces_it():
    base = contrast_texture()
    x = base[None].expand(16, -1, -1, -1).clone()
    x[-8:] = 0.5 + (base - 0.5) * 2
    _, metrics = score_candidates(x, [loop.LoopCandidate(), loop.LoopCandidate(0, 0, 3)], options())
    assert metrics[0, loop.METRICS.index("exposure")] == pytest.approx(0, abs=1e-6)
    assert metrics[0, loop.METRICS.index("contrast")] == pytest.approx(0.12, abs=1e-6)
    # Known .24 -> .18 -> .12 contrast levels in the actual three-frame bridge.
    result = loop.render_cycle(x, loop.LoopCandidate(0, 0, 3))
    window = torch.cat((result[-4:], result[:1]))
    levels = window.std((1, 2), correction=0).mean(-1)
    assert (levels[1:] - levels[:-1]).abs().max() == pytest.approx(0.06, abs=1e-6)
    # Gain-only blending introduces no second contour or displaced object.
    assert metrics[1, loop.METRICS.index("ghosting")] == pytest.approx(0, abs=1e-6)


def test_first_frame_contrast_flash_can_be_trimmed():
    base = contrast_texture()
    x = base[None].expand(40, -1, -1, -1).clone()
    x[0] = 0.5 + (base - 0.5) * 2.5
    result, report = loop.optimize_loop(x, options(max_trim_start=1, max_trim_end=0))
    assert report["selected"] == {"trim_start": 1, "trim_end": 0, "overlap": 0}
    torch.testing.assert_close(result, base[None].expand(39, -1, -1, -1))


def test_tone_matching_only_changes_scoring_and_preserves_alpha():
    base = torch.rand(3, 8, 12, 4, generator=torch.Generator().manual_seed(17))
    tail, head = base.clone(), base.clone()
    tail[..., :3] = 0.4 + (base[..., :3] - 0.5) * 0.6
    head[..., :3] = 0.6 + (base[..., :3] - 0.5) * 0.3
    t_original, h_original = tail.clone(), head.clone()
    t, h = loop._match_pair_tone(tail, head)
    torch.testing.assert_close(t, h, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(t[..., 3], base[..., 3])
    torch.testing.assert_close(tail, t_original)
    torch.testing.assert_close(head, h_original)
    rendered = loop.blend_frames(tail, head, torch.tensor(0.5))
    torch.testing.assert_close(rendered, (t_original + h_original) / 2)


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
def test_same_endpoints_give_same_scores_and_choice_despite_different_middle(quality):
    x = periodic_clip(64)
    other = x.clone()
    other[20:44] = torch.rand(other[20:44].shape, generator=torch.Generator().manual_seed(19))
    opts = options(quality=quality, max_trim_start=3, max_trim_end=4, max_fade=5)
    candidates = loop.enumerate_candidates(len(x), opts)
    first_scores, first_metrics = score_candidates(x, candidates, opts)
    second_scores, second_metrics = score_candidates(other, candidates, opts)
    torch.testing.assert_close(first_scores, second_scores, atol=0, rtol=0)
    torch.testing.assert_close(first_metrics, second_metrics, atol=0, rtol=0)
    _, first = loop.optimize_loop(x, opts)
    _, second = loop.optimize_loop(other, opts)
    assert first["selected"] == second["selected"]
    assert first["metrics"] == second["metrics"]
    assert first["score"] == second["score"]


def test_search_only_creates_boundary_proxies(monkeypatch):
    x = periodic_clip(64)
    opts = options(max_trim_start=3, max_trim_end=4, max_fade=5)
    calls = []
    original = loop._proxy
    def capture(images, size, device, chunk, check_interrupt, indices=None):
        calls.append(indices)
        return original(images, size, device, chunk, check_interrupt, indices)
    monkeypatch.setattr(loop, "_proxy", capture)
    _, report = loop.optimize_loop(x, opts)
    assert len(calls) == 2  # Coarse boundary cache and shortlisted refinement.
    assert calls[0] == loop._boundary_indices(loop.enumerate_candidates(len(x), opts), len(x))
    for indices in calls:
        assert indices is not None and set(indices).isdisjoint(range(20, 44))
    assert report["scoring_scope"] == "candidate boundary windows only"


def test_every_start_end_pair_and_fade_length_is_searched():
    opts = options(max_trim_start=3, max_trim_end=4, max_fade=5)
    actual = set(loop.enumerate_candidates(64, opts))
    expected = {loop.LoopCandidate(a, b, k) for a, b, k in product(range(4), range(5), range(6))}
    assert actual == expected


def test_joint_pair_search_finds_known_clean_start_and_end():
    x = torch.full((40, 8, 12, 3), 0.2)
    x[0], x[1], x[-1] = 0.8, 0.5, 0.6
    result, report = loop.optimize_loop(x, options(max_trim_start=3, max_trim_end=4, max_fade=5))
    assert report["selected"] == {"trim_start": 2, "trim_end": 1, "overlap": 0}
    torch.testing.assert_close(result, torch.full((37, 8, 12, 3), 0.2))


def test_fade_length_adapts_to_endpoint_discontinuity():
    overlaps = []
    for difference in (0.02, 0.08):
        x = torch.full((120, 8, 12, 3), 0.2)
        x[-20:] += difference
        result, report = loop.optimize_loop(x, options(max_trim_start=0, max_trim_end=0, max_fade=12))
        k = report["selected"]["overlap"]
        overlaps.append(k)
        window = torch.cat((result[-k - 2:], result[:2]))
        # Actual sample jumps shrink; the stronger mismatch gets more fade time.
        assert (window[1:] - window[:-1]).abs().max() < difference / 2
    assert 0 < overlaps[0] < overlaps[1] <= 12


def test_motion_direction_change_detected_despite_identical_endpoints():
    base = periodic_clip()
    x = torch.cat((base[:8], base[:8].flip(0)))
    assert torch.equal(x[0], x[-1])
    _, metrics = score_candidates(x, [loop.LoopCandidate()], options())
    # Matching endpoints do not erase a stop/reversal's signed temporal change.
    assert metrics[0, loop.METRICS.index("smoothness")] > 0


def test_detail_regions_reveal_small_object_offset_on_blank_background():
    x = torch.zeros(12, 64, 64, 3)
    x[:6, 24:30, 22:28] = 1
    x[6:, 24:30, 26:32] = 1
    _, m = score_candidates(x, [loop.LoopCandidate()], options())
    global_error = (x[-1] - x[0]).abs().mean()
    assert m[0, 0] > global_error * 2


def quiet_endpoints_with_busy_middle():
    x = torch.full((60, 32, 48, 3), 0.2)
    base = x[0].clone()
    base[12:20, 16:24] = 0.8
    x[:8] = base
    x[-8:] = base.roll(2, dims=1)
    for i in range(8, 52):
        x[i] = base.roll(4 * i, dims=1) * (0.3 if i % 2 else 1)
    return x


def test_busy_middle_cannot_excuse_quiet_endpoint_jump():
    x = quiet_endpoints_with_busy_middle()
    _, metrics = score_candidates(x, [loop.LoopCandidate()], options())
    # The old whole-clip quartile subtracts away this localized discontinuity.
    assert (x[9:52] - x[8:51]).abs().mean() > 0.1
    assert metrics[0, loop.METRICS.index("appearance")] > 0.1
    assert metrics[0, loop.METRICS.index("motion")] > 0.1


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
def test_busy_middle_repair_reduces_actual_transition_jump(quality):
    x = quiet_endpoints_with_busy_middle()
    result, report = loop.optimize_loop(x, options(quality=quality, max_trim_start=0, max_trim_end=0))
    k = report["selected"]["overlap"]
    assert k > 0 and report["selection"] != "unchanged_satisfactory"
    assert report["objective_version"] == 3
    # Independent pixel differences across all bridge links, including repeat.
    window = torch.cat((result[-k - 3:], result[:3]))
    repaired_jump = (window[1:] - window[:-1]).abs().amax()
    original_jump = (x[0] - x[-1]).abs().amax()
    assert repaired_jump < original_jump * 0.5
    # A smoother fade still has ghosting; it must not be advertised as perfect.
    assert report["confidence"] == "low"


def test_two_frame_cut_cannot_borrow_head_motion_tolerance():
    x = torch.full((60, 8, 12, 3), 0.2)
    x[1], x[2], x[3] = 0.24, 0.30, 0.38
    for i in range(8, 52):
        x[i] = 0.1 if i % 2 else 0.9
    _, metrics = score_candidates(x, [loop.LoopCandidate(0, 0, 2)], options())
    # K=2 has no mixed pixels: the only new pair jumps from .2 to .24.
    # Score that cut even though the outgoing, unchanged head motion is faster.
    assert metrics[0, loop.METRICS.index("appearance")] == pytest.approx(0.04, abs=1e-6)
    assert metrics[0, loop.METRICS.index("exposure")] == pytest.approx(0.04, abs=1e-6)
    assert metrics[0, loop.METRICS.index("ghosting")] == 0


def test_moving_objects_long_fades_have_real_ghost_penalty():
    x = torch.zeros(20, 24, 40, 3)
    for i in range(20):
        x[i, 8:16, i:i + 4] = 1
    _, metrics = score_candidates(x, [loop.LoopCandidate(0, 0, 2), loop.LoopCandidate(0, 0, 8)], options())
    assert metrics[0, 4] == 0  # Both K=2 weights are unmixed endpoints.
    assert metrics[1, 4] > 0.05  # Actual overlapping images contain distinct objects.
    _, report = loop.optimize_loop(x, options(max_trim_start=0, max_trim_end=0, max_fade=8))
    assert report["selected"]["overlap"] < 8


def test_joint_grid_zero_fade_duration_and_manual_constraints():
    opts = options(min_retained_percent=60)
    candidates = loop.enumerate_candidates(12, opts)
    assert loop.LoopCandidate() in candidates
    assert loop.LoopCandidate(1, 1, 2) in candidates
    assert any(c.overlap == 0 and c.trim_start > 0 for c in candidates)
    assert all(12 - c.trim_start - c.trim_end - c.overlap >= 8 for c in candidates)
    assert loop.enumerate_candidates(12, replace(opts, manual_trim_start=1, manual_trim_end=1, manual_overlap=2)) == [loop.LoopCandidate(1, 1, 2)]
    with pytest.raises(ValueError, match="constraints|bounds"):
        loop.enumerate_candidates(12, replace(opts, manual_overlap=8))
    with pytest.raises(ValueError, match="constraints|bounds"):
        loop.enumerate_candidates(12, replace(opts, manual_trim_start=4, manual_trim_end=4))
    with pytest.raises(ValueError, match="exceeds"):
        loop.enumerate_candidates(100, replace(opts, max_candidates=2))


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps(fps):
    with pytest.raises(ValueError, match="fps"):
        loop.optimize_loop(torch.zeros(4, 3, 4, 3), options(fps=fps))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_pixels(value):
    x = torch.zeros(4, 3, 4, 3)
    x[2, 1, 1, 1] = value
    with pytest.raises(ValueError, match="pixels|NaN"):
        loop.optimize_loop(x, options())


@pytest.mark.parametrize("x", [torch.zeros(0, 4, 4, 3), torch.zeros(4, 4, 3), torch.zeros(4, 4, 4, 3, dtype=torch.int32)])
def test_invalid_shape_or_dtype(x):
    with pytest.raises(ValueError):
        loop.optimize_loop(x, options())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_noncontiguous_input_shape_dtype_range_and_immutability(dtype):
    x = periodic_clip().to(dtype).transpose(1, 2)
    original = x.clone()
    result, _ = loop.optimize_loop(x, options(manual_overlap=3, manual_trim_start=0, manual_trim_end=0, min_retained_percent=50))
    assert result.shape == (13, 24, 16, 3) and result.dtype == dtype
    assert result.min() >= 0 and result.max() <= 1
    torch.testing.assert_close(x, original)


def test_fps_is_not_retimed_and_temporal_scores_scale():
    x = torch.linspace(0, 1, 12)[:, None, None, None].expand(12, 4, 4, 3)
    _, low = score_candidates(x, [loop.LoopCandidate()], options(fps=24))
    _, high = score_candidates(x, [loop.LoopCandidate()], options(fps=48))
    torch.testing.assert_close(high[0, 0], 2 * low[0, 0])
    torch.testing.assert_close(high[0, 2], 4 * low[0, 2])
    result, report = loop.optimize_loop(x, options(fps=48, manual_overlap=3))
    assert report["fps"] == 48 and report["output_duration_seconds"] == len(result) / 48


def test_low_confidence_and_tie_reports():
    _, poor = loop.optimize_loop(periodic_clip(13), options(max_trim_start=0, max_trim_end=0, max_fade=0, satisfactory_score=0))
    assert poor["confidence"] == "low" and poor["diagnostics"]
    _, tied = loop.optimize_loop(torch.zeros(20, 4, 4, 3), options(tie_tolerance=1))
    assert tied["nearly_tied"] and tied["confidence"] == "uncertain"


def test_missing_cuda_and_comfy_cpu_preference(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, report = loop.select_device("auto")
    assert device.type == "cpu" and report["fallback_reason"]
    with pytest.raises(RuntimeError, match="Forced CUDA"):
        loop.select_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried despite configured CPU"))
    assert loop.select_device("auto", "cpu")[0].type == "cpu"


def test_incompatible_cuda_smoke_fallback_is_narrow(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 3))
    def fail(_):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
    monkeypatch.setattr(loop, "cuda_smoke_test", fail)
    device, report = loop.select_device("auto")
    assert device.type == "cpu" and report["smoke_test"]["passed"] is False
    with pytest.raises(RuntimeError, match="Forced CUDA"):
        loop.select_device("cuda")
    monkeypatch.setattr(loop, "cuda_smoke_test", lambda _: (_ for _ in ()).throw(RuntimeError("shape mismatch bug")))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        loop.select_device("auto")


@pytest.mark.parametrize("forced", [False, True])
def test_bounded_oom_chunk_retries_and_fallback(monkeypatch, forced):
    monkeypatch.setattr(loop, "select_device", lambda *_: (torch.device("cuda:0"), {}))
    monkeypatch.setattr(loop, "_sync", lambda _: None)
    monkeypatch.setattr(loop, "_memory_chunks", lambda *args: (16, 8))
    calls = []
    real_search = loop._search
    def search(images, opts, device, candidates, chunk, render_chunk, *args):
        calls.append((device.type, chunk, render_chunk))
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return real_search(images, opts, device, candidates, chunk, render_chunk, *args)
    monkeypatch.setattr(loop, "_search", search)
    if forced:
        with pytest.raises(RuntimeError, match="exhausted"):
            loop.optimize_loop(torch.zeros(8, 4, 4, 3), options(device="cuda"))
        assert calls == [("cuda", 16, 8), ("cuda", 8, 4), ("cuda", 4, 2)]
    else:
        _, report = loop.optimize_loop(torch.zeros(8, 4, 4, 3), options(device="auto"))
        assert calls == [("cuda", 16, 8), ("cuda", 8, 4), ("cuda", 4, 2), ("cpu", 2, 1)]
        assert len(report["backend"]["attempts"]) == 3 and "OOM" in report["backend"]["fallback_reason"]


def test_cancellation_and_bugs_are_never_fallback(monkeypatch):
    def cancel():
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        loop.optimize_loop(torch.zeros(4, 4, 4, 3), options(), check_interrupt=cancel)
    monkeypatch.setattr(loop, "_search", lambda *args: (_ for _ in ()).throw(RuntimeError("implementation bug")))
    with pytest.raises(RuntimeError, match="implementation bug"):
        loop.optimize_loop(torch.zeros(4, 4, 4, 3), options())


def test_progress_and_node_contract_optional_defaults(monkeypatch):
    management = ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cpu")
    management.intermediate_device = lambda: torch.device("cpu")
    calls = []
    management.throw_exception_if_processing_interrupted = lambda: calls.append("interrupt")
    utils = ModuleType("comfy.utils")
    class Progress:
        def __init__(self, total):
            assert total == 100
        def update_absolute(self, current, total):
            calls.append(current)
    utils.ProgressBar = Progress
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    node = MAIAutoSeamlessLoop()
    result = node.run(torch.zeros(8, 4, 6, 3))
    assert len(result) == 7 and result[1:6] == (24.0, 8, 0, 0, 0)
    assert node.RETURN_NAMES == ("images", "fps", "frame_count", "trim_start", "trim_end", "overlap", "report")
    assert node.CATEGORY == "mAI / Image" and calls[-1] == 100
    assert json.loads(result[-1])["backend"]["selected_device"] == "cpu"


def test_import_does_not_initialize_cuda_and_registration_preserves_nodes():
    code = """
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA touched at import')
torch.cuda.is_available = forbidden
torch.cuda._lazy_init = forbidden
import nodes.auto_seamless_loop
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)
    # Parse registrations without importing ComfyUI-dependent existing nodes.
    import ast
    tree = ast.parse(Path("__init__.py").read_text())
    mappings = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Dict)}
    keys = [ast.literal_eval(key) for key in mappings["NODE_CLASS_MAPPINGS"].keys]
    assert "MAIAutoSeamlessLoop" in keys and "MAIFrameLoopFade" in keys and "MAIVideoLoader" in keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cpu_cuda_ranking_and_render_agreement():
    x = periodic_clip(13)
    candidates = loop.enumerate_candidates(len(x), options())
    cpu, _ = score_candidates(x, candidates, options())
    gpu, _ = score_candidates(x.cuda(), candidates, options())
    torch.testing.assert_close(cpu, gpu.cpu(), atol=2e-5, rtol=2e-4)
    cpu_best, gpu_best = int(cpu.argmin()), int(gpu.argmin())
    assert abs(cpu[cpu_best] - cpu[gpu_best]) <= 2e-5
    y, report = loop.optimize_loop(x, options(device="cuda"))
    z, cpu_report = loop.optimize_loop(x, options(device="cpu"))
    assert report["selected"] == cpu_report["selected"]
    torch.testing.assert_close(y, z, atol=2e-5, rtol=2e-4)
    assert report["backend"]["smoke_test"]["passed"] is True


def test_refinement_windows_match_actual_rendered_cycle():
    x = torch.rand(12, 3, 5, 3, generator=torch.Generator().manual_seed(4))
    for c in [loop.LoopCandidate(1, 1, k) for k in (0, 1, 3, 4)]:
        ix = loop._boundary_indices([c], len(x))
        # Sparse higher-resolution cache must include every referenced source frame.
        lookup = torch.full((len(x),), -1, dtype=torch.long)
        lookup[torch.tensor(ix)] = torch.arange(len(ix))
        full_s, full_m = score_candidates(x, [c], options())
        sparse_s, sparse_m = loop._score_candidates(x[ix], [c], len(x), options(), 2, None, lookup=lookup)
        torch.testing.assert_close(full_s, sparse_s)
        torch.testing.assert_close(full_m, sparse_m)


@pytest.mark.parametrize("candidate", [loop.LoopCandidate(1, 1, k) for k in (0, 1, 3, 4)])
def test_scored_transition_really_includes_rendered_wraparound(monkeypatch, candidate):
    x = torch.rand(12, 3, 5, 3, generator=torch.Generator().manual_seed(9))
    source_edges = loop._edges(x)
    captured = []
    original_edges = loop._edges
    def capture(frames):
        captured.append(frames.clone())
        return original_edges(frames)
    monkeypatch.setattr(loop, "_edges", capture)
    loop._score_batch(x, source_edges, [candidate], len(x), options())
    cycle = loop.render_cycle(x, candidate)
    k = candidate.overlap
    junction = len(x) - candidate.trim_start - candidate.trim_end - 2 * k if k else len(cycle)
    indices = torch.arange(junction - 3, junction + k + 3) % len(cycle)
    torch.testing.assert_close(captured[0][0], cycle[indices])


def test_workflow_wiring_matches_released_sockets_and_widgets():
    workflow = json.loads(Path("examples/auto_seamless_loop.json").read_text())
    api = json.loads(Path("examples/auto_seamless_loop_api.json").read_text())
    node = next(n for n in workflow["nodes"] if n["type"] == "MAIAutoSeamlessLoop")
    assert tuple(output["name"] for output in node["outputs"]) == MAIAutoSeamlessLoop.RETURN_NAMES
    specs = MAIAutoSeamlessLoop.INPUT_TYPES()
    widgets = [name for group in ("required", "optional") for name, spec in specs[group].items() if len(spec) > 1]
    assert len(widgets) == len(node["widgets_values"])
    assert api["2"]["inputs"]["images"] == ["1", 0]
    assert api["2"]["inputs"]["fps"] == ["1", 1]
    assert api["3"]["inputs"]["frames"] == ["2", 0]
    assert api["3"]["inputs"]["frame_rate"] == ["2", 1]
    assert api["3"]["inputs"]["codec"] == "libx264"


def test_inference_autocast_state_is_restored_and_one_pixel_images_work():
    previous = torch.is_inference_mode_enabled()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result, _ = loop.optimize_loop(torch.zeros(8, 1, 1, 3), options())
        assert torch.is_autocast_enabled("cpu")
    assert torch.is_inference_mode_enabled() == previous
    assert result.dtype == torch.float32


def test_full_package_import_registers_new_and_existing_nodes():
    code = """
import importlib.util, pathlib, sys
root = pathlib.Path.cwd()
spec = importlib.util.spec_from_file_location('mai_package_check', root / '__init__.py', submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
assert 'MAIAutoSeamlessLoop' in package.NODE_CLASS_MAPPINGS
assert 'MAIFrameLoopFade' in package.NODE_CLASS_MAPPINGS
assert 'MAIGPUVideoCombine' in package.NODE_CLASS_MAPPINGS
assert package.NODE_DISPLAY_NAME_MAPPINGS['MAIAutoSeamlessLoop'] == 'mAI auto seamless loop'
print(len(package.NODE_CLASS_MAPPINGS), 'nodes registered')
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cpu_proxy_on_cuda_input_uses_cpu_resize(monkeypatch):
    x = periodic_clip().cuda()
    real_interpolate = loop.F.interpolate
    def resize(part, *args, **kwargs):
        assert part.device.type == "cpu"
        return real_interpolate(part, *args, **kwargs)
    monkeypatch.setattr(loop.F, "interpolate", resize)
    result, report = loop.optimize_loop(x, options())
    assert result.device.type == "cpu" and report["backend"]["selected_device"] == "cpu"
    torch.testing.assert_close(result, x.cpu())


def test_cuda_retry_can_succeed_without_fallback(monkeypatch):
    monkeypatch.setattr(loop, "select_device", lambda *_: (torch.device("cuda:0"), {}))
    monkeypatch.setattr(loop, "_sync", lambda _: None)
    monkeypatch.setattr(loop, "_memory_chunks", lambda *args: (16, 8))
    calls = []
    def search(images, opts, device, candidates, chunk, render_chunk, *args):
        calls.append(chunk)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic render OOM")
        return images.clone(), {"timings": {}}
    monkeypatch.setattr(loop, "_search", search)
    _, report = loop.optimize_loop(torch.zeros(8, 4, 4, 3), options(device="auto"))
    assert calls == [16, 8] and report["backend"]["selected_device"] == "cuda:0"
    assert len(report["backend"]["attempts"]) == 1


def test_gpu_only_output_blocks_unsafe_oom_fallback(monkeypatch):
    monkeypatch.setattr(loop, "select_device", lambda *_: (torch.device("cuda:0"), {}))
    monkeypatch.setattr(loop, "_sync", lambda _: None)
    monkeypatch.setattr(loop, "_memory_chunks", lambda *args: (16, 8))
    def fail(*args):
        raise torch.cuda.OutOfMemoryError("synthetic OOM")
    monkeypatch.setattr(loop, "_search", fail)
    with pytest.raises(RuntimeError, match="GPU-only|gpu-only"):
        loop.optimize_loop(torch.zeros(8, 4, 4, 3), options(device="auto"), output_device="cuda")


def test_comfy_cuda_device_query_failure_has_cpu_fallback(monkeypatch):
    management = ModuleType("comfy.model_management")
    def device_failure():
        raise RuntimeError("Found no NVIDIA driver on your system")
    management.get_torch_device = device_failure
    management.intermediate_device = lambda: torch.device("cpu")
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    result = MAIAutoSeamlessLoop().run(torch.zeros(4, 1, 1, 3))
    report = json.loads(result[-1])
    assert report["backend"]["selected_device"] == "cpu"
    assert "device query" in report["backend"]["fallback_reason"]


def test_validation_oom_also_participates_in_bounded_retry(monkeypatch):
    monkeypatch.setattr(loop, "select_device", lambda *_: (torch.device("cuda:0"), {}))
    monkeypatch.setattr(loop, "_sync", lambda _: None)
    monkeypatch.setattr(loop, "_memory_chunks", lambda *args: (16, 8))
    calls = []
    real_validate = loop.validate_images
    def validation(images, chunk, interrupt, device):
        calls.append(chunk)
        if len(calls) <= 3:
            raise torch.cuda.OutOfMemoryError("validation OOM")
        return real_validate(images, chunk, interrupt, device)
    monkeypatch.setattr(loop, "validate_images", validation)
    result, report = loop.optimize_loop(torch.zeros(8, 4, 4, 3), options(device="auto"))
    assert calls == [8, 4, 2, 1]
    assert result.device.type == "cpu" and len(report["backend"]["attempts"]) == 3


def test_seamless_clip_with_long_pause_keeps_useful_motion():
    moving = periodic_clip(8)
    # This cycle completes a translation with a pause inside it. The repeating
    # boundary continues normal motion; trimming away the action would be wasteful.
    x = torch.cat((moving[:4], moving[4:5].expand(16, -1, -1, -1), moving[4:]))
    result, report = loop.optimize_loop(x, options(max_trim_start=4, max_trim_end=4, min_retained_percent=60))
    assert report["selected"] == {"trim_start": 0, "trim_end": 0, "overlap": 0}
    torch.testing.assert_close(result, x)
