"""Blend top and bottom images with a selectable mode and percentage."""

if __package__ == "nodes":
    from utils.image_blend import BLEND_MODES, blend_image_batches
else:
    from ..utils.image_blend import BLEND_MODES, blend_image_batches


class MAIImageBlend:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = (
        "Blends top and bottom images with one mode and percentage across the batch. "
        "A single image repeats across the other batch. Two batches must have matching counts."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_top": ("IMAGE", {"tooltip": "Top image or batch, blended over image_bottom."}),
                "image_bottom": ("IMAGE", {"tooltip": "Base image or batch. A 0% blend shows this input."}),
            },
            "optional": {
                "blend_mode": (list(BLEND_MODES), {"default": "normal"}),
                "blend_percentage": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01,
                    "tooltip": "Applies to the entire batch. 0% keeps the bottom image, and 100% applies the full blend mode.",
                }),
                "blend_list": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "hidden": True,
                    "tooltip": "Percentages saved by an older workflow. Clear this text to use blend_percentage.",
                }),
            },
        }

    def run(self, image_top, image_bottom, blend_list="", blend_mode="normal", blend_percentage=50.0):
        return (blend_image_batches(
            image_top, image_bottom, blend_percentage, blend_mode, blend_list=blend_list,
        ),)
