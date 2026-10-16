try:
    from ..utils.minimax_h3 import (
        DIMENSION_MULTIPLE,
        MEGAPIXEL_OPTIONS,
        RESIZE_MODE_MEGAPIXELS,
        RESIZE_MODE_OPTIONS,
        RESIZE_MODE_SHORT_SIDE_768,
        SHORT_SIDE_TARGET,
        calculate_minimax_h3_dimensions,
        calculate_short_side_dimensions,
    )
except ImportError:
    from utils.minimax_h3 import (
        DIMENSION_MULTIPLE,
        MEGAPIXEL_OPTIONS,
        RESIZE_MODE_MEGAPIXELS,
        RESIZE_MODE_OPTIONS,
        RESIZE_MODE_SHORT_SIDE_768,
        SHORT_SIDE_TARGET,
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
                    {
                        "default": RESIZE_MODE_MEGAPIXELS,
                        "tooltip": "Short side 768 px uses the editable short_side target.",
                    },
                ),
            },
            "optional": {
                "short_side": (
                    "INT",
                    {
                        "default": SHORT_SIDE_TARGET,
                        "min": 1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Shorter-side target in pixels, before rounding to multiple_of. Used only in short-side mode.",
                    },
                ),
                "multiple_of": (
                    "INT",
                    {
                        "default": DIMENSION_MULTIPLE,
                        "min": 1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Round both output dimensions to the nearest multiple of this value. Use 1 for no grid rounding.",
                    },
                ),
            },
        }

    def prepare(
        self,
        image,
        target_megapixels,
        resize_mode=RESIZE_MODE_MEGAPIXELS,
        short_side=SHORT_SIDE_TARGET,
        multiple_of=DIMENSION_MULTIPLE,
    ):
        if image.dim() != 4:
            raise ValueError("image must have shape [batch, height, width, channels]")

        height = image.shape[1]
        width = image.shape[2]
        if resize_mode == RESIZE_MODE_MEGAPIXELS:
            target_width, target_height = calculate_minimax_h3_dimensions(
                width,
                height,
                target_megapixels,
                multiple_of=multiple_of,
            )
        elif resize_mode == RESIZE_MODE_SHORT_SIDE_768:
            target_width, target_height = calculate_short_side_dimensions(
                width, height, short_side=short_side, multiple_of=multiple_of
            )
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
