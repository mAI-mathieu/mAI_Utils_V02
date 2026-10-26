import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from nodes.scheduled_lora import MAIScheduledLora
from utils.lora_schedule import DEFAULT_LORA_SCHEDULE


@pytest.fixture
def native(monkeypatch):
    """Stub only the ComfyUI boundary so these tests need no GPU/runtime."""
    comfy = ModuleType("comfy")
    hooks = ModuleType("comfy.hooks")
    utils = ModuleType("comfy.utils")
    hook = SimpleNamespace(set_keyframes_on_hooks=Mock())
    hooks.create_hook_lora = Mock(return_value=hook)
    hooks.set_hooks_for_conditioning = Mock(side_effect=lambda cond, group, **kw: [cond, group])
    utils.load_torch_file = Mock(return_value={"weights": object()})
    comfy.hooks, comfy.utils = hooks, utils
    folder_paths = SimpleNamespace(
        get_full_path_or_raise=Mock(side_effect=lambda category, name: name),
        get_filename_list=Mock(return_value=["style.safetensors"]),
    )
    adapter = ModuleType("utils.lora_schedule_hooks")
    adapter.LoRAScheduleKeyframes = Mock(return_value=object())
    for name, module in [
        ("comfy", comfy), ("comfy.hooks", hooks), ("comfy.utils", utils),
        ("folder_paths", folder_paths), ("utils.lora_schedule_hooks", adapter),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(hooks=hooks, utils=utils, paths=folder_paths, adapter=adapter, hook=hook)


def test_all_zero_schedule_bypasses_dependencies_and_cloning():
    model, positive, negative = object(), object(), object()
    result = MAIScheduledLora().run(model, positive, negative, "unused", "0:0,100:0", "linear")
    assert result[:3] == (model, positive, negative)
    assert "Step 1/20 | 0% | strength 0" in result[3]


def test_invalid_schedule_and_mode_fail_before_loading(native):
    node = MAIScheduledLora()
    with pytest.raises(ValueError, match="strictly increasing"):
        node.run(object(), [], [], "style", "20:1,10:0", "linear")
    with pytest.raises(ValueError, match="interpolation"):
        node.run(object(), [], [], "style", "0:1", "unknown")
    native.utils.load_torch_file.assert_not_called()


def test_native_hooks_attach_to_both_prompts_and_preserve_model(native):
    model = Mock()
    positive, negative = [[object(), {"other": "positive"}]], [[object(), {"other": "negative"}]]
    result = MAIScheduledLora().run(model, positive, negative, "style", DEFAULT_LORA_SCHEDULE, "linear")
    assert result[0] is model.clone.return_value
    model.clone.assert_called_once_with(disable_dynamic=True)
    native.utils.load_torch_file.assert_called_once_with("style", safe_load=True)
    native.hooks.create_hook_lora.assert_called_once_with(lora=native.utils.load_torch_file.return_value, strength_model=1.0, strength_clip=0.0)
    assert result[1] == [positive, native.hook]
    assert result[2] == [negative, native.hook]
    assert result[3].startswith("LoRA: style\n")
    assert "Step 6/20 | 25% | strength 0.5" in result[3]
    calls = native.hooks.set_hooks_for_conditioning.call_args_list
    assert calls[0].kwargs["cache"] is calls[1].kwargs["cache"]
    assert positive[0][1] == {"other": "positive"}
    assert negative[0][1] == {"other": "negative"}


def test_file_cache_reuses_weights_but_creates_fresh_hooks_and_models(native):
    node, model = MAIScheduledLora(), Mock()
    for name, curve in [("first", "0:1"), ("first", "0:0,100:1"), ("second", "0:1")]:
        node.run(model, [], [], name, curve, "linear")
    assert native.utils.load_torch_file.call_count == 2
    assert native.hooks.create_hook_lora.call_count == 3
    assert model.clone.call_count == 3


def test_missing_file_error_propagates(native):
    native.paths.get_full_path_or_raise.side_effect = FileNotFoundError("Missing LoRA")
    with pytest.raises(FileNotFoundError, match="Missing LoRA"):
        MAIScheduledLora().run(Mock(), [], [], "missing", "0:1", "linear")
    native.utils.load_torch_file.assert_not_called()


def test_schema_uses_installed_loras_and_outputs_match_sampler(native):
    schema = MAIScheduledLora.INPUT_TYPES()
    required = schema["required"]
    assert required["lora_name"][0] == ["style.safetensors"]
    assert required["schedule"][1]["default"] == DEFAULT_LORA_SCHEDULE
    assert required["interpolation"][0] == ["linear", "constant"]
    assert schema["optional"]["report_steps"][1]["default"] == 20
    assert MAIScheduledLora.RETURN_TYPES == ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    assert MAIScheduledLora.RETURN_NAMES == ("model", "positive", "negative", "schedule_text")


def test_report_steps_changes_text_without_changing_hook_schedule(native):
    node, model = MAIScheduledLora(), Mock()
    first = node.run(model, [], [], "style", DEFAULT_LORA_SCHEDULE, "linear", report_steps=10)
    second = node.run(model, [], [], "style", DEFAULT_LORA_SCHEDULE, "linear", report_steps=40)
    assert "Step 2/10 | 10% | strength 1" in first[3]
    assert "Step 11/40 | 25% | strength 0.5" in second[3]
    calls = native.adapter.LoRAScheduleKeyframes.call_args_list
    assert calls[0] == calls[1]
    assert native.utils.load_torch_file.call_count == 1


def test_invalid_report_steps_fail_even_for_bypass():
    with pytest.raises(ValueError, match="report_steps"):
        MAIScheduledLora().run(object(), [], [], "unused", "0:0", "linear", report_steps=0)


@pytest.fixture
def keyframes_type(native):
    # Only the base-class storage is stubbed; execute the real adapter methods.
    class KeyframeGroup:
        def __init__(self):
            self.keyframes = []

        def add(self, keyframe):
            self.keyframes.append(keyframe)

        def reset(self):
            pass

    native.hooks.HookKeyframeGroup = KeyframeGroup
    native.hooks.HookKeyframe = lambda strength, **kw: SimpleNamespace(strength=strength, **kw)
    path = Path(__file__).resolve().parents[1] / "utils" / "lora_schedule_hooks.py"
    spec = importlib.util.spec_from_file_location("utils._tested_lora_schedule_hooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LoRAScheduleKeyframes


class SamplerSigmas(list):
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


def test_adapter_tracks_strength_without_advancing_on_repeated_calls(keyframes_type):
    curve = keyframes_type(((0, 1), (10, 1), (40, 0), (100, 0)), "linear")
    options = {"sample_sigmas": SamplerSigmas(range(20, -1, -1))}
    assert not curve.prepare_current_keyframe(20, options)
    assert not curve.prepare_current_keyframe(18, options)
    assert curve.prepare_current_keyframe(15, options)
    assert curve.strength == 0.5
    assert not curve.prepare_current_keyframe(15, options)
    assert curve.prepare_current_keyframe(12, options)
    assert curve.strength == 0
    assert not curve.prepare_current_keyframe(1, options)


def test_adapter_reset_and_clone_keep_the_custom_schedule(keyframes_type):
    curve = keyframes_type(((0, -1), (100, 1)), "linear")
    options = {"sample_sigmas": SamplerSigmas([100, 1, 0])}
    curve.prepare_current_keyframe(1, options)
    assert curve.strength == 0
    clone = curve.clone()
    assert type(clone) is type(curve)
    assert clone.points == curve.points
    assert clone.strength == -1
    curve.initialize_timesteps(object())
    assert curve.strength == -1
    assert curve._sample_sigmas is None
    # A new pass, including reduced denoise, gets its own step boundaries.
    assert curve.prepare_current_keyframe(0.5, {"sample_sigmas": SamplerSigmas([1, 0.5, 0])})
    assert curve.strength == 0


def test_adapter_constant_boundaries_and_missing_sampler_schedule(keyframes_type):
    curve = keyframes_type(((0, 1), (50, 0)), "constant")
    options = {"sample_sigmas": SamplerSigmas([10, 1, 0])}
    assert not curve.prepare_current_keyframe(5, options)
    assert curve.prepare_current_keyframe(1, options)
    assert curve.strength == 0
    with pytest.raises(ValueError, match="sample_sigmas"):
        curve.prepare_current_keyframe(0, {})


def test_pack_registration_preserves_existing_nodes():
    # Other registered image/video nodes need PyTorch, but no loaded models.
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("mai_schedule_registration", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    try:
        spec.loader.exec_module(package)
        assert package.NODE_CLASS_MAPPINGS["MAIScheduledLora"].__name__ == "MAIScheduledLora"
        assert package.NODE_DISPLAY_NAME_MAPPINGS["MAIScheduledLora"] == "mAI scheduled LoRA"
        assert "MAIConditionalLora" in package.NODE_CLASS_MAPPINGS
        assert "MAIAutoSeamlessLoop" in package.NODE_CLASS_MAPPINGS
    finally:
        for name in list(sys.modules):
            if name == spec.name or name.startswith(spec.name + "."):
                del sys.modules[name]
