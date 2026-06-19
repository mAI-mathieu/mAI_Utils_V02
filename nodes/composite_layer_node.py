try:
    from ..utils.image_composite import composite_layer
except ImportError:
    from utils.image_composite import composite_layer


class MAICompositeLayer:
    CATEGORY = "mAI / Image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "composite"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                "y": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "anchor": (["top_left", "center"], {"default": "top_left"}),
                "fit_mode": (["none", "contain", "cover"], {"default": "none"}),
            },
            "optional": {
                "base_mask": ("MASK",),
                "layer_mask": ("MASK",),
            },
        }

    def composite(
        self,
        base_image,
        layer_image,
        x,
        y,
        scale,
        opacity,
        anchor,
        fit_mode,
        base_mask=None,
        layer_mask=None,
    ):
        return composite_layer(
            base_image,
            layer_image,
            x,
            y,
            scale,
            opacity=opacity,
            anchor=anchor,
            fit_mode=fit_mode,
            base_mask=base_mask,
            layer_mask=layer_mask,
        )
