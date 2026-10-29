"""ComfyUI wrapper for automatic source-preserving edit compositing."""

try:
    from ..utils.image_preserve_source import preserve_source_image
    from ..utils.masked_image_composite import BLEND_MODES
except ImportError:
    from utils.image_preserve_source import preserve_source_image
    from utils.masked_image_composite import BLEND_MODES


class MAIImagePreserveSource:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "change_mask")
    DESCRIPTION = (
        "Keep the original image except where a substantial edit is detected. "
        "Compare aligned images at pixel and blurred scales, then blend their "
        "join with feather, multiband, or content-aware seams."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "edited_image": ("IMAGE",),
                "difference_threshold": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Ignore RGB differences below this value. Raise to preserve more source.",
                }),
                "transition_width": ("FLOAT", {
                    "default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Smooth transition above the threshold; zero makes a hard selection.",
                }),
                "comparison_blur": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 32.0, "step": 0.1,
                    "tooltip": "Gaussian spread in pixels for broad-change detection. Zero compares raw pixels.",
                }),
                "detail_sensitivity": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 favors broad changes; 1 detects all raw pixel changes, including texture drift.",
                }),
                "mask_expand": ("INT", {"default": 2, "min": 0, "max": 128,
                    "tooltip": "Expand detected edits by this many pixels before feathering."}),
                "mask_blur": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 32.0, "step": 0.1,
                    "tooltip": "Gaussian mask feather spread in pixels. Zero disables feathering.",
                }),
                "blend_mode": (list(BLEND_MODES), {"default": "multiband",
                    "tooltip": "Content-aware modes optimize the seam inside the gray mask band."}),
                "multiband_levels": ("INT", {"default": 5, "min": 1, "max": 10,
                    "tooltip": "Pyramid levels in multiband modes. More levels blend broader transitions."}),
            },
            "optional": {
                "edit_mask": ("MASK", {
                    "tooltip": "Optional allowed edit area. Black always protects the original; white allows detection.",
                }),
            },
        }

    def run(
        self, original_image, edited_image, difference_threshold=0.08,
        transition_width=0.04, comparison_blur=2.0, detail_sensitivity=0.25,
        mask_expand=2, mask_blur=2.0, blend_mode="multiband", multiband_levels=5,
        edit_mask=None,
    ):
        return preserve_source_image(
            original_image, edited_image, difference_threshold, transition_width,
            comparison_blur, detail_sensitivity, mask_expand, mask_blur,
            blend_mode, multiband_levels, edit_mask,
        )
