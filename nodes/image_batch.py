"""Batch available images, including optional outputs from mAI image gate."""

try:
    from ..utils.image_batch import batch_connected_images
except ImportError:
    from utils.image_batch import batch_connected_images


class MAIImageBatch:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Batches available images in socket order, skipping unconnected inputs and "
        "None from disabled mAI image gates. Later images are resized to the first "
        "available image's width and height. Images must share channels, dtype and "
        "device. Outputs None when no images are available."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                f"image_{number}": ("IMAGE", {
                    "tooltip": "Optional image or batch. Missing images and disabled mAI image gates are skipped.",
                })
                for number in range(1, 9)
            },
        }

    def run(
        self, image_1=None, image_2=None, image_3=None, image_4=None,
        image_5=None, image_6=None, image_7=None, image_8=None,
    ):
        return (batch_connected_images((
            image_1, image_2, image_3, image_4,
            image_5, image_6, image_7, image_8,
        )),)
