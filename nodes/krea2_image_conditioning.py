try:
    from ..utils.krea2_conditioning import (
        CONDITIONING_MODES, DEFAULT_INSTRUCTION, DETAIL_INSTRUCTIONS,
        create_krea2_conditioning,
    )
except ImportError:
    from utils.krea2_conditioning import (
        CONDITIONING_MODES, DEFAULT_INSTRUCTION, DETAIL_INSTRUCTIONS,
        create_krea2_conditioning,
    )


class MAIKrea2ImageConditioning:
    CATEGORY = "mAI / Conditioning"
    FUNCTION = "process"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "caption")
    DESCRIPTION = (
        "Reuses the native Krea 2 Qwen3-VL-4B CLIP to caption reference images "
        "and build text, image-aware, or joint conditioning. Batches produce "
        "one shared conditioning from all references. In vl_only mode, caption "
        "generation is skipped and the caption output is an empty string."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "clip": ("CLIP",),
            "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION}),
            "conditioning_mode": (list(CONDITIONING_MODES), {"default": "text_plus_vl"}),
            "detail_level": (list(DETAIL_INSTRUCTIONS), {"default": "high"}),
            "caption_max_new_tokens": ("INT", {"default": 192, "min": 32, "max": 1024}),
        }}

    def process(
        self, image, clip, instruction=DEFAULT_INSTRUCTION,
        conditioning_mode="text_plus_vl", detail_level="high", caption_max_new_tokens=192,
    ):
        return create_krea2_conditioning(
            clip, image, instruction, conditioning_mode, detail_level, caption_max_new_tokens,
        )
