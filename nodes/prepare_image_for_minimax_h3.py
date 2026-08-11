try:
    from ..utils.minimax_h3 import MEGAPIXEL_OPTIONS, calculate_minimax_h3_dimensions
except ImportError:
    from utils.minimax_h3 import MEGAPIXEL_OPTIONS, calculate_minimax_h3_dimensions


class MAIPrepareImageForMinimaxH3:
    CATEGORY = "mAI / Image"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": (list(MEGAPIXEL_OPTIONS), {"default": "1.0 MP"}),
            }
        }

    def prepare(self, image, target_megapixels):
        if image.dim() != 4:
            raise ValueError("image must have shape [batch, height, width, channels]")

        height = image.shape[1]
        width = image.shape[2]
        target_width, target_height = calculate_minimax_h3_dimensions(
            width,
            height,
            target_megapixels,
        )

        if width == target_width and height == target_height:
            return (image,)

        import comfy.utils

        channels_first = image.movedim(-1, 1)
        resized = comfy.utils.common_upscale(
            channels_first,
            target_width,
            target_height,
            "lanczos",
            "disabled",
        )
        return (resized.movedim(1, -1),)
