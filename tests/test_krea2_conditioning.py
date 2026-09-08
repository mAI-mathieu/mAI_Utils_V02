import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nodes.krea2_image_conditioning import MAIKrea2ImageConditioning
from utils.krea2_conditioning import (
    DEFAULT_INSTRUCTION,
    build_image_conditioning_template,
    build_reconstruction_instruction,
    clean_caption,
    create_krea2_conditioning,
    validate_krea2_clip,
)


@pytest.fixture
def clip(monkeypatch):
    """Exercise adapter calls without importing ComfyUI or allocating model weights."""
    native = ModuleType("comfy.text_encoders.krea2")
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")

    class InterruptProcessingException(BaseException):
        pass

    management.InterruptProcessingException = InterruptProcessingException
    management.interrupted = False

    def check_interrupt():
        if management.interrupted:
            management.interrupted = False
            raise InterruptProcessingException()

    management.throw_exception_if_processing_interrupted = check_interrupt
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)

    class Krea2TEModel:
        def __init__(self):
            self.reset_count = 0
            self.execution_device = None

        def reset_clip_options(self):
            self.reset_count += 1
            self.execution_device = None

    class Krea2Qwen3VLClipModel:
        def __init__(self):
            self.transformer = SimpleNamespace(
                model=SimpleNamespace(graph_dynamic_vbar_blocks=True),
            )

        def generate(self):
            pass

    class Krea2Tokenizer:
        pass

    native.Krea2TEModel = Krea2TEModel
    native.Krea2Qwen3VLClipModel = Krea2Qwen3VLClipModel
    native.Krea2Tokenizer = Krea2Tokenizer
    native.KREA2_TEMPLATE = (
        "<|im_start|>system\nNative Krea description instruction.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
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
            self.management = management

        def tokenize(self, text, **kwargs):
            tokens = {"text": text, **kwargs}
            self.token_calls.append(tokens)
            return tokens

        def generate(self, tokens, **kwargs):
            assert not torch.is_grad_enabled()
            assert not self.cond_stage_model.qwen3vl_4b.transformer.model.graph_dynamic_vbar_blocks
            self.generation_calls.append((tokens, kwargs))
            return [17, 18]

        def decode(self, ids):
            assert ids == [17, 18]
            return self.caption_text

        def encode_from_tokens_scheduled(self, tokens):
            assert not self.cond_stage_model.qwen3vl_4b.transformer.model.graph_dynamic_vbar_blocks
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
    if mode == "vl_only":
        assert caption == ""
        assert not clip.generation_calls
    else:
        assert caption == clip.caption_text
        generation_tokens, generation_options = clip.generation_calls[0]
        assert torch.equal(generation_tokens["image"], image)
        assert generation_tokens["thinking"] is False
        assert "llama_template" not in generation_tokens  # Keep native caption generation template.
        assert generation_options == {"do_sample": False, "max_length": 64}
        assert DEFAULT_INSTRUCTION in generation_tokens["text"]
    assert clip.encoded_tokens["text"] == ("" if mode == "vl_only" else caption)
    if mode == "text_only":
        assert "image" not in clip.encoded_tokens
        assert "llama_template" not in clip.encoded_tokens
    else:
        assert torch.equal(clip.encoded_tokens["image"], image)
        template = clip.encoded_tokens["llama_template"]
        assert template.startswith("<|im_start|>system\n")
        assert "<|im_start|>user\n<|vision_start|>" in template
    assert "thinking" not in clip.encoded_tokens  # Keep Krea 2 conditioning default.


@pytest.mark.parametrize("batch_size", [1, 3])
def test_vl_only_has_no_caption_dependency_and_encodes_all_references(clip, batch_size):
    # Generation/decoding may be unavailable or broken without blocking VL mode.
    clip.generate = None
    clip.decode = None
    clip.cond_stage_model.qwen3vl_4b.generate = None
    image = torch.rand(batch_size, 8, 8, 3)
    conditioning, caption = create_krea2_conditioning(
        clip, image, conditioning_mode="vl_only", instruction=None,
        detail_level=None, caption_max_new_tokens=None,
    )
    assert caption == ""
    assert conditioning is clip.conditioning
    assert not clip.generation_calls
    assert len(clip.token_calls) == 1
    assert clip.encoded_tokens["text"] == ""
    assert torch.equal(clip.encoded_tokens["image"], image)
    assert clip.encoded_tokens["llama_template"].count("<|image_pad|>") == batch_size
    assert clip.cond_stage_model.reset_count == 1


def test_vl_only_stop_propagates_and_restores_encoder_options(clip):
    interrupted = clip.management.InterruptProcessingException()

    def fail(tokens):
        raise interrupted

    clip.encode_from_tokens_scheduled = fail
    with pytest.raises(clip.management.InterruptProcessingException) as caught:
        create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3), conditioning_mode="vl_only")
    assert caught.value is interrupted
    assert not clip.generation_calls
    assert clip.cond_stage_model.reset_count == 1
    assert clip.cond_stage_model.qwen3vl_4b.transformer.model.graph_dynamic_vbar_blocks


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
    assert clip.encoded_tokens["llama_template"].count("<|image_pad|>") == 2


@pytest.mark.parametrize("image_count", [1, 2, 4])
def test_conditioning_prefix_ends_before_every_image_placeholder(clip, image_count):
    native_template = sys.modules["comfy.text_encoders.krea2"].KREA2_TEMPLATE
    template = build_image_conditioning_template(native_template, image_count)
    prefix, suffix = native_template.split("{}")
    assert template.startswith(prefix)
    assert template.endswith("{}" + suffix)
    assert template.count("<|image_pad|>") == image_count
    # Native Krea stripping locates the second im_start. Its index must not
    # depend on image token expansion or caption length (the old template did).
    turns = template.split("<|im_start|>")
    assert turns[1].startswith("system\n")
    assert turns[2].startswith("user\n<|vision_start|>")
    assert turns[3].startswith("assistant\n")
    assert prefix.count("<|im_start|>") == 2
    assert "<|image_pad|>" not in prefix
    # Formatting must leave caption braces intact.
    assert "A sign saying {OPEN}" in template.format("A sign saying {OPEN}")


@pytest.mark.parametrize("template", [
    None, "{}", "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\n{}<|im_start|>user\n{}",
])
def test_rejects_templates_that_break_native_prefix_stripping(template):
    with pytest.raises(ValueError, match="native system/user template"):
        build_image_conditioning_template(template, 1)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_rejects_invalid_reference_counts(clip, count):
    native_template = sys.modules["comfy.text_encoders.krea2"].KREA2_TEMPLATE
    with pytest.raises(ValueError, match="positive integer"):
        build_image_conditioning_template(native_template, count)


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
    assert len(clip.generation_calls) == 2


@pytest.mark.parametrize("empty", ["", "  ", "<think>unfinished", "<think>done</think>"])
def test_empty_response_retries_same_image_and_preserves_instruction(clip, empty, caplog):
    responses = iter((empty, "A blue vase beside a window."))
    clip.decode = lambda ids: next(responses)
    image = torch.rand(1, 8, 8, 3)
    conditioning, caption = create_krea2_conditioning(
        clip, image, instruction="Focus on object placement.", caption_max_new_tokens=512,
    )
    first, retry = clip.generation_calls
    assert retry[0]["text"].startswith(first[0]["text"])
    assert "Focus on object placement." in retry[0]["text"]
    assert retry[0]["text"] != first[0]["text"]
    assert torch.equal(retry[0]["image"], image)
    assert retry[0]["thinking"] is False
    assert retry[1] == first[1] == {"do_sample": False, "max_length": 512}
    assert caption == "A blue vase beside a window."
    assert clip.encoded_tokens["text"] == caption
    assert conditioning is clip.conditioning
    assert clip.cond_stage_model.reset_count == 3
    assert "retrying once" in caplog.text


@pytest.mark.parametrize("count, hint", [(1, "ended before"), (32, "reached its token budget")])
def test_failed_retry_reports_counts_and_distinguishes_early_stop_from_budget(clip, count, hint):
    clip.generate = lambda *args, **kwargs: [151645] * count
    clip.decode = lambda ids: ""
    with pytest.raises(RuntimeError, match="after 2 attempts") as caught:
        create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3), caption_max_new_tokens=32)
    assert f"[{count}, {count}]" in str(caught.value)
    assert hint in str(caught.value)
    assert clip.encoded_tokens is None


def test_batch_retries_only_failed_image_without_reusing_previous_caption(clip):
    responses = iter(("A vase.", "", "A chair."))
    clip.decode = lambda ids: next(responses)
    image = torch.stack((torch.zeros(8, 8, 3), torch.ones(8, 8, 3)))
    _, caption = create_krea2_conditioning(clip, image)
    assert caption == "Image 1: A vase.\n\nImage 2: A chair."
    assert len(clip.generation_calls) == 3
    assert torch.equal(clip.generation_calls[2][0]["image"], image[1:2])


def test_batch_failure_identifies_image_and_stops_before_conditioning(clip):
    responses = iter(("A vase.", "", ""))
    clip.decode = lambda ids: next(responses)
    with pytest.raises(RuntimeError, match="image 2 after 2 attempts"):
        create_krea2_conditioning(clip, torch.zeros(3, 8, 8, 3))
    assert len(clip.generation_calls) == 3
    assert clip.encoded_tokens is None


@pytest.mark.parametrize("failure", ["decode", "runtime", "interrupt"])
def test_retry_does_not_swallow_native_errors_or_interruptions(clip, failure):
    original = clip.generate

    def retry_fails(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(clip.generation_calls) == 2:
            if failure == "interrupt":
                raise clip.management.InterruptProcessingException()
            if failure == "runtime":
                raise RuntimeError("native generation failed")
        return result

    clip.generate = retry_fails
    responses = iter(("", None))
    clip.decode = lambda ids: next(responses)
    expected = clip.management.InterruptProcessingException if failure == "interrupt" else RuntimeError
    with pytest.raises(expected):
        create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3))
    assert len(clip.generation_calls) == 2
    assert clip.encoded_tokens is None
    assert clip.cond_stage_model.reset_count == 2
    assert clip.cond_stage_model.qwen3vl_4b.transformer.model.graph_dynamic_vbar_blocks


def test_invalid_decode_result_does_not_trigger_retry(clip):
    clip.decode = lambda ids: None
    with pytest.raises(RuntimeError, match="decoding did not return text"):
        create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3))
    assert len(clip.generation_calls) == 1


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
    assert clip.cond_stage_model.reset_count == 1


@pytest.mark.parametrize("stage", ["generate", "encode_from_tokens_scheduled"])
def test_stop_restores_options_and_allows_rerun(clip, stage):
    original = getattr(clip, stage)
    interrupted = clip.management.InterruptProcessingException()
    decoder = clip.cond_stage_model.qwen3vl_4b.transformer.model

    def interrupted_operation(*args, **kwargs):
        clip.cond_stage_model.execution_device = "test execution device"
        assert decoder.graph_dynamic_vbar_blocks is False
        raise interrupted

    setattr(clip, stage, interrupted_operation)
    with pytest.raises(clip.management.InterruptProcessingException) as caught:
        create_krea2_conditioning(clip, torch.zeros(2, 8, 8, 3))
    assert caught.value is interrupted
    assert decoder.graph_dynamic_vbar_blocks is True
    assert clip.cond_stage_model.execution_device is None
    assert clip.encoded_tokens is None

    setattr(clip, stage, original)
    result, caption = create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3))
    assert result is clip.conditioning
    assert caption == clip.caption_text
    assert decoder.graph_dynamic_vbar_blocks is True


@pytest.mark.parametrize("has_graph_flag", [True, False])
def test_decoders_without_enabled_graph_capture_keep_their_configuration(clip, has_graph_flag):
    decoder = clip.cond_stage_model.qwen3vl_4b.transformer.model
    if has_graph_flag:
        decoder.graph_dynamic_vbar_blocks = False
    else:
        del decoder.graph_dynamic_vbar_blocks
    # These versions have no active graph path to assert inside the doubles.
    clip.generate = lambda *args, **kwargs: [17, 18]
    clip.encode_from_tokens_scheduled = lambda *args: clip.conditioning
    create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3))
    assert hasattr(decoder, "graph_dynamic_vbar_blocks") == has_graph_flag
    if has_graph_flag:
        assert decoder.graph_dynamic_vbar_blocks is False


def test_stop_before_generation_does_no_model_work(clip):
    clip.management.interrupted = True
    with pytest.raises(clip.management.InterruptProcessingException):
        create_krea2_conditioning(clip, torch.zeros(1, 8, 8, 3))
    assert not clip.generation_calls
    assert clip.encoded_tokens is None
    assert clip.cond_stage_model.execution_device is None


@pytest.mark.parametrize("stage", ["generate", "encode_from_tokens_scheduled"])
def test_stop_at_end_of_native_call_discards_outputs(clip, stage):
    original = getattr(clip, stage)

    def request_stop(*args, **kwargs):
        result = original(*args, **kwargs)
        clip.management.interrupted = True
        return result

    setattr(clip, stage, request_stop)
    with pytest.raises(clip.management.InterruptProcessingException):
        create_krea2_conditioning(clip, torch.zeros(2, 8, 8, 3))
    if stage == "generate":
        assert len(clip.generation_calls) == 1
        assert clip.encoded_tokens is None
    assert clip.cond_stage_model.execution_device is None


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
