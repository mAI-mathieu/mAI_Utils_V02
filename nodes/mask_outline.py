try:
    from ..utils.mask_outline import draw_mask_outline
except ImportError:
    from utils.mask_outline import draw_mask_outline


class MAIMaskOutline:
    CATEGORY = "mAI / Mask"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw_outline"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "#00FF00"}),
                "thickness": (
                    "INT",
                    {"default": 10, "min": 1, "max": 4096, "step": 1},
                ),
                "padding": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
                "threshold": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1},
                ),
            }
        }

    def draw_outline(self, image, mask, color, thickness, padding, threshold):
        outlined = draw_mask_outline(
            image=image,
            mask=mask,
            color=color,
            thickness=thickness,
            padding=padding,
            threshold=threshold,
        )
        return (outlined,)
