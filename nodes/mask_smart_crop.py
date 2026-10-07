"""ComfyUI wrapper for native-pixel mask crops."""

try:
    from ..utils.mask_smart_crop import CONTEXT_TYPE, smart_crop
except ImportError:
    from utils.mask_smart_crop import CONTEXT_TYPE, smart_crop


class MAIMaskSmartCrop:
    CATEGORY = "mAI / Mask"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", CONTEXT_TYPE, "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "full_crop_mask", "crop_context",
                    "crop_x", "crop_y", "crop_width", "crop_height", "scale")
    DESCRIPTION = (
        "Expand around the mask using native image pixels. Resize only when the padded "
        "mask cannot fit the target, or when allow_upscale permits it for a small image. "
        "Full crop mask marks the exact source rectangle. Connect crop_context to Smart Stitch. "
        "Scalar geometry outputs describe the first image in a batch."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "Source-size mask; one mask broadcasts across an image batch."}),
                "output_width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "output_height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
            },
            "optional": {
                "mask_padding": ("INT", {"default": 0, "min": 0, "max": 16384,
                                         "tooltip": "Source pixels of context added to the mask bbox before fitting."}),
                "mask_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Pixels strictly above this value define the bbox. Soft mask values are retained."}),
                "allow_upscale": ("BOOLEAN", {"default": False,
                                               "tooltip": "Allow a smaller native rectangle to be upscaled for small source images, if the full padded mask fits."}),
                "edge_mode": (["shift", "pad"], {"default": "shift",
                                                 "tooltip": "shift maximizes real pixels. pad retains mask centering with replicated image edges and zero mask padding."}),
            },
        }

    def run(self, image, mask, output_width=1024, output_height=1024,
            mask_padding=0, mask_threshold=0.01, allow_upscale=False, edge_mode="shift"):
        return smart_crop(image, mask, output_width, output_height, mask_padding,
                          mask_threshold, allow_upscale, edge_mode)
