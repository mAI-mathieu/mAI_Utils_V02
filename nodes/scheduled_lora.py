"""Apply a configurable LoRA strength timeline during diffusion sampling."""

if __package__ == "nodes":
    from utils.lora_schedule import DEFAULT_LORA_SCHEDULE, parse_lora_schedule
else:
    from ..utils.lora_schedule import DEFAULT_LORA_SCHEDULE, parse_lora_schedule


class MAIScheduledLora:
    CATEGORY = "mAI / Utils"
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    DESCRIPTION = (
        "Controls diffusion-model LoRA strength throughout sampling. Enter one "
        "percent:strength point per line. Linear fades between points; constant "
        "holds until the next point. Connect all three outputs to the sampler. "
        "Percentages refer to the steps executed in each sampler run."
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
                "tooltip": "percent:strength, one per line (or comma-separated). Example: 0:1, 10:1, 40:0, 100:0. Percentages: 0–100; strengths: -100–100.",
            }),
            "interpolation": (["linear", "constant"], {
                "tooltip": "Linear fades between points. Constant holds each strength until the next point.",
            }),
        }}

    def run(self, model, positive, negative, lora_name, schedule, interpolation):
        points = parse_lora_schedule(schedule)
        if interpolation not in ("linear", "constant"):
            raise ValueError(f"Unknown LoRA schedule interpolation: {interpolation}")
        if all(strength == 0 for _, strength in points):
            return (model, positive, negative)

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
        return (patched_model, positive, negative)
