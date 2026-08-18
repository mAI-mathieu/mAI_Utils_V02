try:
    from ..utils.mask_bounding_box import create_bounding_box_mask
except ImportError:
    from utils.mask_bounding_box import create_bounding_box_mask


class MAIMaskBoundingBox:
    CATEGORY = "mAI / Mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    def create_mask(self, mask):
        return (create_bounding_box_mask(mask),)
