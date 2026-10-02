from contextlib import nullcontext
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from utils.gpu_memory import (
    NVENC_HEADROOM_BYTES, gpu_memory_scope, prepare_gpu_memory,
    release_gpu_cache, video_memory_requirements,
)


@pytest.fixture
def fake_cuda(monkeypatch):
    events = []
    state = {"free": 10_000, "cached": 0}
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty",)))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (state["free"], 100_000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: state["cached"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    management = SimpleNamespace(free_memory=lambda size, device: events.append(("offload", size, device)))
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    return events, state, management


def test_cpu_never_touches_cuda_or_comfy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA touched")
    monkeypatch.setattr(torch.cuda, "device", forbidden)
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(free_memory=forbidden))
    release_gpu_cache("cpu")
    prepare_gpu_memory("cpu", 100_000)
    with gpu_memory_scope("cpu", 100_000):
        pass


def test_cache_cleared_before_headroom_check_and_after_success(fake_cuda):
    events, _, _ = fake_cuda
    with gpu_memory_scope("cuda:2", 100):
        events.append(("work",))
    assert [event[0] for event in events] == ["sync", "empty", "work", "sync", "empty"]
    assert events[0][1] == torch.device("cuda:2")


def test_only_low_headroom_offloads_target_models_accounting_for_cached_bytes(fake_cuda):
    events, state, _ = fake_cuda
    state.update(free=20, cached=50)
    prepare_gpu_memory("cuda:2", 100)
    assert events[2] == ("offload", 150, torch.device("cuda:2"))
    assert [event[0] for event in events] == ["sync", "empty", "offload", "sync", "empty"]


def test_force_offload_even_when_driver_reports_plenty_of_free_memory(fake_cuda):
    events, _, _ = fake_cuda
    prepare_gpu_memory("cuda:1", 100, force_offload=True)
    assert events[2] == ("offload", 1e30, torch.device("cuda:1"))


def test_unindexed_cuda_uses_explicit_current_device_for_comfy_offload(fake_cuda,monkeypatch):
    events, _, _ = fake_cuda
    monkeypatch.setattr(torch.cuda,"current_device",lambda:3)
    with gpu_memory_scope("cuda",force_offload=True):
        pass
    assert events[2] == ("offload",1e30,torch.device("cuda:3"))
    assert all(event[1] == torch.device("cuda:3") for event in events if event[0] == "sync")


@pytest.mark.parametrize("error", [ValueError("invalid"), InterruptedError("cancelled")])
def test_cleanup_on_errors_releases_failed_worker_traceback_locals(fake_cuda, error):
    events, _, _ = fake_cuda
    references = []
    def worker():
        scratch = torch.zeros(1)
        references.append(weakref.ref(scratch))
        raise error
    with pytest.raises(type(error)) as caught:
        with gpu_memory_scope("cuda:0"):
            worker()
    assert caught.value is error
    assert references[0]() is None
    assert [event[0] for event in events] == ["sync", "empty", "sync", "empty"]


def test_cleanup_if_preparation_fails(fake_cuda):
    events, _, management = fake_cuda
    def failed(*args):
        raise RuntimeError("offload failed")
    management.free_memory = failed
    with pytest.raises(RuntimeError, match="offload failed"):
        with gpu_memory_scope("cuda:0", force_offload=True):
            pytest.fail("must not execute")
    assert [event[0] for event in events] == ["sync", "empty", "sync", "empty"]


def test_standalone_cuda_without_comfy_still_releases_cache(fake_cuda, monkeypatch):
    events, _, _ = fake_cuda
    monkeypatch.delitem(sys.modules, "comfy.model_management")
    with gpu_memory_scope("cuda:0", 100_000):
        pass
    assert [event[0] for event in events] == ["sync", "empty", "sync", "empty"]


def test_cpu_frames_prepare_comfy_encoder_and_explicit_gpu(fake_cuda, monkeypatch):
    _, _, management = fake_cuda
    management.get_torch_device = lambda: torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert video_memory_requirements("cpu", -1, 100) == {torch.device("cuda:0"): NVENC_HEADROOM_BYTES}
    assert video_memory_requirements("cpu", 2, 100) == {torch.device("cuda:2"): NVENC_HEADROOM_BYTES}
    assert video_memory_requirements("cuda:2", 1, 100) == {
        torch.device("cuda:2"): 100, torch.device("cuda:1"): NVENC_HEADROOM_BYTES,
    }


def test_same_gpu_budgets_combine_and_standalone_cpu_does_not_initialize_cuda(fake_cuda, monkeypatch):
    _, _, management = fake_cuda
    management.get_torch_device = lambda: torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert video_memory_requirements("cuda:1", 1, 100) == {torch.device("cuda:1"): NVENC_HEADROOM_BYTES + 100}
    monkeypatch.delitem(sys.modules, "comfy.model_management")
    assert video_memory_requirements("cpu", -1, 100) == {}
