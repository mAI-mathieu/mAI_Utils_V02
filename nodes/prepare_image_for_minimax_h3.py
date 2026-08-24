try:
    from ..utils.minimax_h3 import (
        MEGAPIXEL_OPTIONS,
        RESIZE_MODE_MEGAPIXELS,
        RESIZE_MODE_OPTIONS,
        RESIZE_MODE_SHORT_SIDE_768,
        calculate_minimax_h3_dimensions,
        calculate_short_side_dimensions,
    )
except ImportError:
    from utils.minimax_h3 import (
        MEGAPIXEL_OPTIONS,
        RESIZE_MODE_MEGAPIXELS,
        RESIZE_MODE_OPTIONS,
        RESIZE_MODE_SHORT_SIDE_768,
        calculate_minimax_h3_dimensions,
        calculate_short_side_dimensions,
    )


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
                "resize_mode": (
                    list(RESIZE_MODE_OPTIONS),
                    {"default": RESIZE_MODE_MEGAPIXELS},
                ),
            }
        }

    def prepare(self, image, target_megapixels, resize_mode=RESIZE_MODE_MEGAPIXELS):
        if image.dim() != 4:
            raise ValueError("image must have shape [batch, height, width, channels]")

        height = image.shape[1]
        width = image.shape[2]
        if resize_mode == RESIZE_MODE_MEGAPIXELS:
            target_width, target_height = calculate_minimax_h3_dimensions(
                width,
                height,
                target_megapixels,
            )
        elif resize_mode == RESIZE_MODE_SHORT_SIDE_768:
            target_width, target_height = calculate_short_side_dimensions(width, height)
        else:
            raise ValueError(f"Unknown resize mode: {resize_mode}")

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
