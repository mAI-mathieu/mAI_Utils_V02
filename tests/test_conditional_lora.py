import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import nodes
from nodes.conditional_lora import MAIConditionalLora


@pytest.fixture
def native_loader(monkeypatch):
    loader = Mock()
    loader.load_lora.return_value = (object(), object())
    factory = Mock(return_value=loader)
    monkeypatch.setattr(nodes, "LoraLoader", factory, raising=False)
    return factory, loader


@pytest.mark.parametrize(("prompt", "strength", "triggers"), [
    ("a dog", 1.0, "cat"),
    ("a cat", 0.0, "cat"),
    ("a cat", 1.0, ""),
    ("", 1.0, "cat"),
])
def test_bypass_preserves_inputs_without_loading(native_loader, prompt, strength, triggers):
    factory, loader = native_loader
    model, clip = object(), object()
    result = MAIConditionalLora().run(model, "unused.safetensors", prompt, strength, triggers, clip=clip)
    assert result[0] is model
    assert result[1] is clip
    assert result[2] == prompt
    factory.assert_not_called()
    loader.load_lora.assert_not_called()


@pytest.mark.parametrize("strength", [0.75, -0.5])
def test_match_applies_same_strength_once_and_preserves_prompt(native_loader, strength):
    factory, loader = native_loader
    model, clip = object(), object()
    prompt = "a CAT and a dog"
    result = MAIConditionalLora().run(model, "style.safetensors", prompt, strength, "cat,dog", clip=clip)
    assert result == (*loader.load_lora.return_value, prompt)
    factory.assert_called_once_with()
    loader.load_lora.assert_called_once_with(model, clip, "style.safetensors", strength, strength)


def test_toggle_and_file_change_reuse_loader_but_patch_incoming_models(native_loader):
    factory, loader = native_loader
    node = MAIConditionalLora()
    model, clip = object(), object()
    node.run(model, "first.safetensors", "cat", 1.0, "cat", clip=clip)
    assert node.run(model, "first.safetensors", "dog", 1.0, "cat", clip=clip) == (model, clip, "dog")
    node.run(model, "second.safetensors", "cat", 0.5, "cat", clip=clip)
    factory.assert_called_once_with()
    assert loader.load_lora.call_count == 2
    loader.load_lora.assert_called_with(model, clip, "second.safetensors", 0.5, 0.5)


def test_loading_error_is_not_hidden(native_loader):
    _, loader = native_loader
    loader.load_lora.side_effect = FileNotFoundError("Missing LoRA")
    with pytest.raises(FileNotFoundError, match="Missing LoRA"):
        MAIConditionalLora().run(object(), "missing.safetensors", "cat", 1.0, "cat", clip=object())


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -101.0, 101.0])
def test_invalid_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="LoRA strength"):
        MAIConditionalLora().run(object(), "style.safetensors", "cat", strength, "cat", clip=object())


def test_schema_uses_installed_loras_and_stable_outputs(monkeypatch):
    get_filenames = Mock(return_value=["style.safetensors"])
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_filename_list=get_filenames))
    schema = MAIConditionalLora.INPUT_TYPES()
    inputs = schema["required"]
    assert "clip" not in inputs
    assert schema["optional"]["clip"][0] == "CLIP"
    get_filenames.assert_called_once_with("loras")
    assert inputs["lora_name"][0] == ["style.safetensors"]
    assert inputs["prompt"] == ("STRING", {"forceInput": True})
    assert MAIConditionalLora.RETURN_TYPES == ("MODEL", "CLIP", "STRING")
    assert MAIConditionalLora.RETURN_NAMES == ("model", "clip", "prompt")


def test_missing_clip_applies_model_only(native_loader):
    _, loader = native_loader
    model, patched_model = object(), object()
    loader.load_lora.return_value = (patched_model, None)
    result = MAIConditionalLora().run(model, "style.safetensors", "cat", 0.75, "cat")
    loader.load_lora.assert_called_once_with(model, None, "style.safetensors", 0.75, 0.0)
    assert result == (patched_model, None, "cat")


@pytest.mark.parametrize(("prompt", "strength"), [("dog", 1.0), ("cat", 0.0)])
def test_missing_clip_bypass_preserves_model(native_loader, prompt, strength):
    factory, _ = native_loader
    model = object()
    result = MAIConditionalLora().run(model, "unused.safetensors", prompt, strength, "cat")
    assert result == (model, None, prompt)
    factory.assert_not_called()
