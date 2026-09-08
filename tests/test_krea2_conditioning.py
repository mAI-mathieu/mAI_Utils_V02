import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nodes.krea2_image_conditioning import MAIKrea2ImageConditioning
from utils.krea2_conditioning import (
    DEFAULT_INSTRUCTION,
    build_reconstruction_instruction,
    clean_caption,
    create_krea2_conditioning,
    validate_krea2_clip,
)


@pytest.fixture
def clip(monkeypatch):
    """Exercise adapter calls without importing ComfyUI or allocating model weights."""
    native = ModuleType("comfy.text_encoders.krea2")

    class Krea2TEModel:
        pass

    class Krea2Qwen3VLClipModel:
        def generate(self):
            pass

    class Krea2Tokenizer:
        pass

    native.Krea2TEModel = Krea2TEModel
    native.Krea2Qwen3VLClipModel = Krea2Qwen3VLClipModel
    native.Krea2Tokenizer = Krea2Tokenizer
    monkeypatch.setitem(sys.modules, native.__name__, native)

    class NativeClip:
        def __init__(self):
            self.cond_stage_model = Krea2TEModel()
            self.cond_stage_model.qwen3vl_4b = Krea2Qwen3VLClipModel()
            self.tokenizer = Krea2Tokenizer()
            self.token_calls = []
            self.generation_calls = []
            self.encoded_tokens = None
            self.conditioning = [[torch.ones(1, 2, 12 * 2560), {"metadata": object()}]]
            self.caption_text = "A red cup at the left of a wooden table."

        def tokenize(self, text, **kwargs):
            tokens = {"text": text, **kwargs}
            self.token_calls.append(tokens)
            return tokens

        def generate(self, tokens, **kwargs):
            assert not torch.is_grad_enabled()
            self.generation_calls.append((tokens, kwargs))
            return [17, 18]

        def decode(self, ids):
            assert ids == [17, 18]
            return self.caption_text

        def encode_from_tokens_scheduled(self, tokens):
            self.encoded_tokens = tokens
            return self.conditioning

    return NativeClip()


@pytest.mark.parametrize("mode", ["text_only", "vl_only", "text_plus_vl"])
def test_modes_pass_real_images_and_caption_to_the_correct_native_path(clip, mode):
    image = torch.rand(1, 16, 16, 3)
    conditioning, caption = MAIKrea2ImageConditioning().process(
        image, clip, conditioning_mode=mode, caption_max_new_tokens=64,
    )

    assert conditioning is clip.conditioning  # Preserve native tensors and metadata.
    assert caption == clip.caption_text
    generation_tokens, generation_options = clip.generation_calls[0]
    assert torch.equal(generation_tokens["image"], image)
    assert generation_tokens["thinking"] is False
    assert generation_options == {"do_sample": False, "max_length": 64}
    assert DEFAULT_INSTRUCTION in generation_tokens["text"]
    assert clip.encoded_tokens["text"] == ("" if mode == "vl_only" else caption)
    if mode == "text_only":
        assert "image" not in clip.encoded_tokens
    else:
        assert torch.equal(clip.encoded_tokens["image"], image)
    assert "thinking" not in clip.encoded_tokens  # Keep Krea 2 conditioning default.


def test_batch_captions_are_sequential_and_all_references_reach_conditioning(clip):
    image = torch.stack((torch.zeros(16, 16, 3), torch.ones(16, 16, 3)))
    conditioning, caption = create_krea2_conditioning(clip, image)
    assert len(clip.generation_calls) == 2
    for index, (tokens, _) in enumerate(clip.generation_calls):
        assert torch.equal(tokens["image"], image[index:index + 1])
    assert caption == f"Image 1: {clip.caption_text}\n\nImage 2: {clip.caption_text}"
    assert clip.encoded_tokens["text"] == caption
    assert torch.equal(clip.encoded_tokens["image"], image)
    assert conditioning is clip.conditioning


def test_rgba_alpha_is_ignored_without_changing_rgb(clip):
    image = torch.rand(1, 16, 16, 4)
    create_krea2_conditioning(clip, image)
    assert torch.equal(clip.encoded_tokens["image"], image[..., :3])


@pytest.mark.parametrize("image", [
    None, torch.zeros(0, 8, 8, 3), torch.zeros(1, 0, 8, 3),
    torch.zeros(8, 8, 3), torch.zeros(1, 8, 8, 1),
    torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
    torch.full((1, 8, 8, 3), float("nan")),
    torch.full((1, 8, 8, 3), float("inf")),
    torch.full((1, 8, 8, 3), -0.1), torch.full((1, 8, 8, 3), 1.1),
])
def test_invalid_images_fail_before_model_execution(clip, image):
    with pytest.raises(ValueError, match="image"):
        create_krea2_conditioning(clip, image)
    assert not clip.generation_calls


@pytest.mark.parametrize("options", [
    {"conditioning_mode": "unknown"}, {"detail_level": "unknown"},
    {"instruction": None}, {"caption_max_new_tokens": 31},
    {"caption_max_new_tokens": 1025}, {"caption_max_new_tokens": 32.5},
    {"caption_max_new_tokens": True},
])
def test_invalid_controls_fail_before_generation(clip, options):
    with pytest.raises(ValueError):
        create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3), **options)
    assert not clip.generation_calls


def test_instruction_override_blank_fallback_and_detail_levels():
    assert DEFAULT_INSTRUCTION in build_reconstruction_instruction("  ", "high")
    prompts = [build_reconstruction_instruction("Focus on layout.", level)
               for level in ("low", "medium", "high", "extreme")]
    assert len(set(prompts)) == 4
    assert all(text.startswith("Focus on layout.") for text in prompts)
    assert all(DEFAULT_INSTRUCTION not in text for text in prompts)


def test_clean_caption_removes_reasoning():
    assert clean_caption("<think>analysis\nmore</think> A cup. ") == "A cup."


@pytest.mark.parametrize("text", ["", "  ", "<think>unfinished", "<think>done</think>"])
def test_empty_caption_fails_without_encoding_fallback(clip, text):
    clip.caption_text = text
    with pytest.raises(RuntimeError, match="empty reconstruction caption"):
        create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3))
    assert clip.encoded_tokens is None


def test_wrong_encoder_and_tokenizer_fail_clearly(clip):
    with pytest.raises(ValueError, match="Krea 2 Qwen3-VL-4B"):
        validate_krea2_clip(SimpleNamespace(cond_stage_model=object()))
    clip.tokenizer = object()
    with pytest.raises(ValueError, match="CLIPLoader type 'krea2'"):
        validate_krea2_clip(clip)


def test_missing_native_generation_fails_clearly(clip):
    clip.generate = None
    with pytest.raises(RuntimeError, match="lacks the native generate API"):
        validate_krea2_clip(clip)


def test_native_errors_propagate_without_fallback(clip):
    def fail(*args, **kwargs):
        raise RuntimeError("vision weights unavailable")
    clip.generate = fail
    with pytest.raises(RuntimeError, match="vision weights unavailable"):
        create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3))
    assert clip.encoded_tokens is None


def test_pack_registration_and_socket_contract(monkeypatch):
    repo_dir = Path(__file__).resolve().parents[1]
    module_name = "test_mai_krea2_pack"
    spec = importlib.util.spec_from_file_location(
        module_name, repo_dir / "__init__.py", submodule_search_locations=[str(repo_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, package)
    spec.loader.exec_module(package)
    key = "MAIKrea2ImageConditioning"
    assert package.NODE_CLASS_MAPPINGS[key].__name__ == key
    assert package.NODE_DISPLAY_NAME_MAPPINGS[key] == "mAI Krea2 image conditioning"
    assert MAIKrea2ImageConditioning.RETURN_TYPES == ("CONDITIONING", "STRING")
    assert MAIKrea2ImageConditioning.RETURN_NAMES == ("conditioning", "caption")
    assert MAIKrea2ImageConditioning.CATEGORY == "mAI / Conditioning"
    assert "MAIBackgroundLightingMatch" in package.NODE_CLASS_MAPPINGS
    assert "MAIVideoLoader" in package.NODE_CLASS_MAPPINGS
