"""Small adapter around ComfyUI's native Krea 2 tokenizer and CLIP APIs."""

import re

import torch


DEFAULT_INSTRUCTION = (
    "Describe this image extremely precisely for reconstruction. Preserve subject "
    "identity, pose, facial expression, framing, camera viewpoint, composition, "
    "object positions, proportions, colors, lighting direction, materials, textures, "
    "and background layout. Focus on details that help recreate the same image "
    "rather than reinterpret it."
)
CONDITIONING_MODES = ("text_only", "text_plus_vl", "vl_only")
DETAIL_INSTRUCTIONS = {
    "low": "Be concise: prioritize the subject, framing, major colors and lighting.",
    "medium": "Include composition, pose, clothing, object placement and background.",
    "high": "Describe fine appearance details, spatial relationships, materials and light.",
    "extreme": (
        "Be exhaustive within the token budget: include subtle textures, small objects, "
        "relative proportions, facial details, shadows and background geometry."
    ),
}


def build_reconstruction_instruction(instruction, detail_level):
    if detail_level not in DETAIL_INSTRUCTIONS:
        raise ValueError(f"Unknown detail level: {detail_level}")
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a string.")
    return "\n\n".join((
        instruction.strip() or DEFAULT_INSTRUCTION,
        DETAIL_INSTRUCTIONS[detail_level],
        "Return only the visual reconstruction caption, without reasoning or a preamble. "
        "Describe visible evidence; do not invent obscured details.",
    ))


def clean_caption(text):
    if not isinstance(text, str):
        raise RuntimeError("Krea 2 caption decoding did not return text.")
    # Also discard unfinished reasoning when the generation budget was exhausted.
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
    if not text:
        raise RuntimeError(
            "Krea 2 generated an empty reconstruction caption. Try increasing "
            "caption_max_new_tokens or changing the instruction."
        )
    return text


def build_image_conditioning_template(krea_template, image_count):
    """Put image placeholders after Krea's system+user prefix, never before it."""
    if (
        not isinstance(krea_template, str)
        or krea_template.count("{}") != 1
        or not krea_template.startswith("<|im_start|>system\n")
        or "<|im_start|>user\n{}" not in krea_template
    ):
        raise ValueError("Krea 2 conditioning requires its native system/user template.")
    if not isinstance(image_count, int) or isinstance(image_count, bool) or image_count < 1:
        raise ValueError("image_count must be a positive integer.")
    vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
    return krea_template.replace("{}", vision_block * image_count + "{}", 1)


def validate_krea2_clip(clip):
    # Lazy import keeps the pack usable on installations without Krea 2 support.
    try:
        from comfy.text_encoders.krea2 import (
            Krea2Qwen3VLClipModel,
            Krea2TEModel,
            Krea2Tokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "mAI Krea2 image conditioning requires ComfyUI's native Krea 2 "
            "Qwen3-VL support. Update ComfyUI to a build that includes it."
        ) from exc

    model = getattr(clip, "cond_stage_model", None)
    if not (
        isinstance(model, Krea2TEModel)
        and isinstance(getattr(model, "qwen3vl_4b", None), Krea2Qwen3VLClipModel)
        and isinstance(getattr(clip, "tokenizer", None), Krea2Tokenizer)
    ):
        raise ValueError(
            "This node expects the Krea 2 Qwen3-VL-4B encoder / CLIP. "
            "Load the full Qwen3-VL-4B encoder with CLIPLoader type 'krea2'; "
            "a generic Qwen-VL or other CLIP does not provide Krea 2 conditioning."
        )
    for name in ("tokenize", "generate", "decode", "encode_from_tokens_scheduled"):
        if not callable(getattr(clip, name, None)):
            raise RuntimeError(
                f"The connected Krea 2 CLIP lacks the native {name} API. "
                "Update ComfyUI; this node cannot substitute another encoder."
            )
    if not callable(getattr(model.qwen3vl_4b, "generate", None)):
        raise RuntimeError("This Krea 2 encoder does not support native caption generation.")


def validate_image(image):
    if not torch.is_tensor(image):
        raise ValueError("image must be a ComfyUI IMAGE tensor [B, H, W, C].")
    if image.ndim != 4 or any(size == 0 for size in image.shape):
        raise ValueError("image must be a non-empty ComfyUI IMAGE tensor [B, H, W, C].")
    if image.shape[-1] not in (3, 4) or not torch.is_floating_point(image):
        raise ValueError("image must contain floating-point RGB or RGBA pixels in [0, 1].")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image pixels must be finite and in [0, 1].")
    # Keep BHWC and native image preprocessing. Alpha does not condition the encoder.
    return image[..., :3]


def _native_clip_call(clip, operation, *args, **kwargs):
    """Avoid decoder graph capture and restore options on every exit path."""
    from comfy import model_management

    transformer = clip.cond_stage_model.qwen3vl_4b.transformer
    decoder = transformer.model
    graph_enabled = getattr(decoder, "graph_dynamic_vbar_blocks", False)
    try:
        model_management.throw_exception_if_processing_interrupted()
        # Some native Llama-family decoders capture fixed-KV decode operations.
        # Interrupted capture/replay can leave graph-owned allocations for the
        # executor's later GC. Use native eager decoding for this node only.
        if graph_enabled:
            decoder.graph_dynamic_vbar_blocks = False
        result = operation(*args, **kwargs)
        model_management.throw_exception_if_processing_interrupted()
        return result
    finally:
        if graph_enabled:
            decoder.graph_dynamic_vbar_blocks = graph_enabled
        # CLIP.generate changes execution options without a native finally block.
        # Preserve native exceptions. Do not clear tracebacks, collect garbage,
        # force-unload models or empty CUDA caches on the cancellation path.
        clip.cond_stage_model.reset_clip_options()


@torch.inference_mode()
def create_krea2_conditioning(
    clip, image, instruction=DEFAULT_INSTRUCTION, conditioning_mode="text_plus_vl",
    detail_level="high", caption_max_new_tokens=192,
):
    if conditioning_mode not in CONDITIONING_MODES:
        raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")
    if (
        not isinstance(caption_max_new_tokens, int)
        or isinstance(caption_max_new_tokens, bool)
        or not 32 <= caption_max_new_tokens <= 1024
    ):
        raise ValueError("caption_max_new_tokens must be an integer between 32 and 1024.")
    prompt = build_reconstruction_instruction(instruction, detail_level)
    image = validate_image(image)
    validate_krea2_clip(clip)

    captions = []
    for index in range(image.shape[0]):
        # Match native Generate Text: image template, no reasoning, native KV cache.
        tokens = clip.tokenize(prompt, image=image[index:index + 1], thinking=False)
        generated = _native_clip_call(
            clip, clip.generate, tokens, do_sample=False, max_length=caption_max_new_tokens,
        )
        captions.append(clean_caption(clip.decode(generated)))
    caption = captions[0] if len(captions) == 1 else "\n\n".join(
        f"Image {index + 1}: {text}" for index, text in enumerate(captions)
    )

    text = "" if conditioning_mode == "vl_only" else caption
    if conditioning_mode == "text_only":
        tokens = clip.tokenize(text)
    else:
        from comfy.text_encoders.krea2 import KREA2_TEMPLATE

        # Passing image alone selects Qwen's user/assistant generation template.
        # Krea's prefix stripper assumes the SECOND im_start is the user turn;
        # with the generic template it instead cuts at the assistant token index,
        # which is also wrong after an image placeholder expands into many tokens.
        # Keep Krea's system/user prefix so stripping ends BEFORE image expansion.
        template = build_image_conditioning_template(KREA2_TEMPLATE, image.shape[0])
        tokens = clip.tokenize(text, image=image, llama_template=template)
    # This API applies Krea 2's 12-layer taps,
    # prefix stripping, flattening and standard conditioning metadata unchanged.
    conditioning = _native_clip_call(clip, clip.encode_from_tokens_scheduled, tokens)
    return conditioning, caption
