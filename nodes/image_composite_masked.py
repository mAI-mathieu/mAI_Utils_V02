"""Composite two aligned images with seam processing confined to a soft mask."""

try:
    from ..utils.masked_image_composite import BLEND_MODES, composite_with_source_mask
except ImportError:
    from utils.masked_image_composite import BLEND_MODES, composite_with_source_mask


class MAIImageCompositeMasked:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Composite source_image onto destination_image using source_mask. "
        "White takes the source exactly; black keeps the destination exactly. "
        "Only gray mask pixels receive the selected feather/seam blending. "
        "Images and mask must have matching spatial dimensions."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination_image": ("IMAGE", {"tooltip": "Background/base image. Kept wherever source_mask is black (0)."}),
                "source_image": ("IMAGE", {"tooltip": "Image placed onto destination_image. Taken exactly wherever source_mask is white (1). Same size and channels as destination_image."}),
                "source_mask": ("MASK", {"tooltip": "Applies to source_image: white = take source, black = keep destination, gray (0 < mask < 1) = feather area. No extra feather is generated."}),
                "blend_mode": (list(BLEND_MODES), {"default": "feather", "tooltip": "Separate modes: normal feather, content-aware graph-cut seam, hard cut at mask 0.5, or multiband blending. Changes stay inside gray mask pixels."}),
            },
            "optional": {
                "multiband_levels": ("INT", {"default": 5, "min": 1, "max": 10,
                                            "tooltip": "Pyramid levels, used only by multiband. More levels blend larger features; 1 equals normal feather."}),
            },
        }

    def run(self, destination_image, source_image, source_mask,
            blend_mode="feather", multiband_levels=5):
        return (composite_with_source_mask(destination_image, source_image, source_mask,
                                           blend_mode, multiband_levels),)
