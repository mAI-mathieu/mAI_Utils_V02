try:
    from ..utils.cinematic_post import FLOAT_CONTROLS, PRESETS, cinematic_post
except ImportError:
    from utils.cinematic_post import FLOAT_CONTROLS, PRESETS, cinematic_post


# Explicit user-requested identifier; keep stable for saved workflows.
class mAI_CinematicPost:
    CATEGORY = "mAI / Image"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Restrained cinematic tone, color, optics and grain. Presets set a base "
        "look; sliders trim it relative to Subtle Film defaults. Off / Neutral "
        "bypasses processing. All controls remain visible in this pack."
    )

    @classmethod
    def INPUT_TYPES(cls):
        controls = {
            "enabled": ("BOOLEAN", {"default": True}),
            "preset": (list(PRESETS), {"default": "Subtle Film"}),
            "advanced_mode": ("BOOLEAN", {
                "default": False,
                "tooltip": "UI hint only: this pack keeps all controls visible in both modes.",
            }),
        }
        for name, (default, minimum, maximum, step) in FLOAT_CONTROLS.items():
            controls[name] = ("FLOAT", {
                "default": default, "min": minimum, "max": maximum, "step": step,
            })
            if name == "grain_chroma":
                controls["grain_seed"] = ("INT", {"default": 0, "min": 0, "max": 2**63 - 1})
                controls["grain_animation_safe"] = ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "False repeats a fixed grain pattern; true uses seed + batch index. "
                        "Neither tracks motion."
                    ),
                })
        return {"required": {"image": ("IMAGE",), **controls},
                "optional": {"subject_mask": ("MASK",)}}

    def process(self, image, subject_mask=None, **controls):
        return (cinematic_post(image, subject_mask=subject_mask, **controls),)
