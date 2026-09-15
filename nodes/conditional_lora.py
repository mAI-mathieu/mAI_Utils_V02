"""Apply a LoRA only when the supplied prompt contains a configured trigger."""

import math

if __package__ == "nodes":
    from utils.lora_triggers import prompt_matches_triggers
else:
    from ..utils.lora_triggers import prompt_matches_triggers


class MAIConditionalLora:
    CATEGORY = "mAI / Utils"
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "prompt")
    DESCRIPTION = (
        "Applies the selected LoRA when any trigger word or phrase occurs in "
        "the prompt. Separate triggers with commas or newlines. Matching is "
        "case-insensitive and uses whole words. Strength affects MODEL and, "
        "when connected, the optional CLIP."
    )

    def __init__(self):
        self._loader = None

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "prompt": ("STRING", {"forceInput": True}),
            "strength": ("FLOAT", {
                "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                "tooltip": "LoRA strength for MODEL and optional CLIP. Zero bypasses the LoRA.",
            }),
            "trigger_words": ("STRING", {
                "default": "", "multiline": True,
                "tooltip": "Any matching trigger activates the LoRA. Separate words or phrases with commas or newlines. Blank disables it.",
            }),
        }, "optional": {
            "clip": ("CLIP", {"tooltip": "Optional text encoder. Leave disconnected for a model-only LoRA."}),
        }}

    def run(self, model, lora_name, prompt, strength, trigger_words, clip=None):
        if not math.isfinite(strength) or not -100.0 <= strength <= 100.0:
            raise ValueError("LoRA strength must be a finite number between -100 and 100.")

        if strength == 0 or not prompt_matches_triggers(prompt, trigger_words):
            return (model, clip, prompt)

        # Reuse ComfyUI's native loader, including its file cache and metadata
        # handling. Import lazily so bypass never loads LoRA dependencies/files.
        if self._loader is None:
            from nodes import LoraLoader

            self._loader = LoraLoader()

        patched_model, patched_clip = self._loader.load_lora(
            model, clip, lora_name, strength, strength if clip is not None else 0.0,
        )
        return (patched_model, patched_clip, prompt)
