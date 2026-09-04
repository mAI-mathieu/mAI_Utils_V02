try:
    from ..utils.background_lighting import match_background_lighting
except ImportError:
    from utils.background_lighting import match_background_lighting


class MAIBackgroundLightingMatch:
    CATEGORY = "mAI / Image"
    FUNCTION = "match_lighting"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "edited_image": ("IMAGE",),
            },
            "optional": {
                "edit_mask": ("MASK",),
            },
        }

    def match_lighting(self, original_image, edited_image, edit_mask=None):
        corrected = match_background_lighting(
            original_image,
            edited_image,
            edit_mask,
        )
        return (corrected,)
