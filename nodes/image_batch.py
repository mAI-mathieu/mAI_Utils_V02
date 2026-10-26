"""Batch available images, including optional outputs from mAI image gate."""

try:
    from ..utils.image_batch import RESIZE_METHODS, batch_connected_images
except ImportError:
    from utils.image_batch import RESIZE_METHODS, batch_connected_images


class MAIImageBatch:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Batches available images in socket order, skipping unconnected inputs and "
        "None from disabled mAI image gates. Later images are resized to the first "
        "available image's width and height using stretch, preserve ratio crop, or "
        "preserve ratio pad. Images must share channels, dtype and "
        "device. Outputs None when no images are available."
    )

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"image_{number}": ("IMAGE", {
                "tooltip": "Optional image or batch. Missing images and disabled mAI image gates are skipped.",
            })
            for number in range(1, 9)
        }
        optional["resize_method"] = (list(RESIZE_METHODS), {
            "default": "stretch",
            "tooltip": "Match the first available image: stretch, scale and center-crop, or scale and center-pad with black.",
        })
        return {
            "required": {},
            "optional": optional,
        }

    def run(
        self, image_1=None, image_2=None, image_3=None, image_4=None,
        image_5=None, image_6=None, image_7=None, image_8=None,
        resize_method="stretch",
    ):
        return (batch_connected_images((
            image_1, image_2, image_3, image_4,
            image_5, image_6, image_7, image_8,
        ), resize_method=resize_method),)
