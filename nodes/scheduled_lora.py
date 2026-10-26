"""Apply a configurable LoRA strength timeline during diffusion sampling."""

if __package__ == "nodes":
    from utils.lora_schedule import (
        DEFAULT_LORA_SCHEDULE, INTERPOLATION_METHODS,
        format_lora_schedule_report, parse_lora_schedule,
    )
else:
    from ..utils.lora_schedule import (
        DEFAULT_LORA_SCHEDULE, INTERPOLATION_METHODS,
        format_lora_schedule_report, parse_lora_schedule,
    )


class MAIScheduledLora:
    CATEGORY = "mAI / Utils"
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "schedule_text")
    DESCRIPTION = (
        "Controls diffusion-model LoRA strength throughout sampling. Enter one "
        "percent%:strength point per line, e.g. 25%:0.5. Linear fades at a steady rate; "
        "ease_in, ease_out and ease_in_out shape each fade; constant holds until the "
        "next point. Connect model, positive and negative to the sampler. "
        "Percentages refer to the steps executed in each sampler run. The schedule_text "
        "output previews strength per step using report_steps; set it to the sampler's executed step count."
    )

    def __init__(self):
        self._loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "schedule": ("STRING", {
                "default": DEFAULT_LORA_SCHEDULE, "multiline": True,
                "tooltip": "Progress % : LoRA strength, one per line (or comma-separated). Example: 0%:1, 10%:1, 40%:0, 100%:0. 10%:1 means strength 1 at 10% of sampling. Percentages: 0–100; strengths: -100–100.",
            }),
            "interpolation": (list(INTERPOLATION_METHODS), {
                "tooltip": "Linear: steady fade. Constant: hold then switch. Ease in: starts slowly, finishes quickly. Ease out: starts quickly, finishes slowly. Ease in/out: slow start and finish. Easing applies separately between each pair of points, for fades in or out.",
            }),
        }, "optional": {
            "report_steps": ("INT", {
                "default": 20, "min": 1, "max": 10000,
                "tooltip": "Executed sampler steps used only for the schedule_text preview. Match the sampler's actual step count (after denoise/start/end trimming). Does not change sampling. Can share an INT input with the sampler's steps.",
            }),
        }}

    def run(self, model, positive, negative, lora_name, schedule, interpolation, report_steps=20):
        points = parse_lora_schedule(schedule)
        if interpolation not in INTERPOLATION_METHODS:
            raise ValueError(f"Unknown LoRA schedule interpolation: {interpolation}")
        schedule_text = f"LoRA: {lora_name}\n" + format_lora_schedule_report(
            points, interpolation, report_steps,
        )
        if all(strength == 0 for _, strength in points):
            return (model, positive, negative, schedule_text)

        import comfy.hooks
        import comfy.utils
        import folder_paths

        if __package__ == "nodes":
            from utils.lora_schedule_hooks import LoRAScheduleKeyframes
        else:
            from ..utils.lora_schedule_hooks import LoRAScheduleKeyframes

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self._loaded_lora is None or self._loaded_lora[0] != path:
            self._loaded_lora = None
            self._loaded_lora = (path, comfy.utils.load_torch_file(path, safe_load=True))

        hooks = comfy.hooks.create_hook_lora(
            lora=self._loaded_lora[1], strength_model=1.0, strength_clip=0.0,
        )
        hooks.set_keyframes_on_hooks(LoRAScheduleKeyframes(points, interpolation))
        # Native weight hooks require the ordinary patcher rather than dynamic
        # VRAM patching. The incoming model and its other patches stay intact.
        patched_model = model.clone(disable_dynamic=True)
        cache = {}
        positive = comfy.hooks.set_hooks_for_conditioning(positive, hooks, cache=cache)
        negative = comfy.hooks.set_hooks_for_conditioning(negative, hooks, cache=cache)
        return (patched_model, positive, negative, schedule_text)
