"""ComfyUI wrapper for reversing Smart Crop transforms."""

try:
    from ..utils.mask_smart_crop import CONTEXT_TYPE
    from ..utils.mask_smart_stitch import COMPOSITE_MODES, smart_stitch
except ImportError:
    from utils.mask_smart_crop import CONTEXT_TYPE
    from utils.mask_smart_stitch import COMPOSITE_MODES, smart_stitch


class MAIMaskSmartStitch:
    CATEGORY = "mAI / Mask"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Restore a generated Smart Crop to its exact source location and original resolution. "
        "The default mask_feather mode uses the saved native semantic mask when mask is absent. "
        "Temporary padding is discarded; full_crop replaces exactly the full_crop_mask rectangle."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE", {"tooltip": "Same dimensions, channels and batch order as cropped_image."}),
                "crop_context": (CONTEXT_TYPE,),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional override at cropped_image resolution, e.g. cropped_mask. Otherwise uses the saved source mask."}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 1024,
                                    "tooltip": "Gaussian feather radius in original source pixels; only used in mask_feather mode. Clipped to the valid crop rectangle."}),
                "composite_mode": (list(COMPOSITE_MODES), {"default": "mask_feather",
                                                          "tooltip": "full_crop replaces the rectangle; mask_only uses the soft semantic mask; mask_feather optionally softens its edges."}),
            },
        }

    def run(self, original_image, generated_image, crop_context, mask=None,
            feather=0, composite_mode="mask_feather"):
        return (smart_stitch(original_image, generated_image, crop_context,
                             mask, feather, composite_mode),)
