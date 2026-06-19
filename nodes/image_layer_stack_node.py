try:
    from ..utils.image_composite import compose_layer_stack
except ImportError:
    from utils.image_composite import compose_layer_stack


class MAIImageLayerStack:
    CATEGORY = "mAI / Image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compose"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for layer_number in range(1, 9):
            prefix = f"layer_{layer_number}"
            optional.update(
                {
                    f"{prefix}_image": ("IMAGE",),
                    f"{prefix}_mask": ("MASK",),
                    f"{prefix}_x": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                    f"{prefix}_y": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                    f"{prefix}_scale": (
                        "FLOAT",
                        {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01},
                    ),
                    f"{prefix}_opacity": (
                        "FLOAT",
                        {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                    ),
                    f"{prefix}_anchor": (["top_left", "center"], {"default": "top_left"}),
                    f"{prefix}_fit_mode": (["none", "contain", "cover"], {"default": "none"}),
                }
            )

        return {
            "required": {
                "background": ("IMAGE",),
            },
            "optional": optional,
        }

    def compose(self, background, **kwargs):
        layers = []
        for layer_number in range(1, 9):
            prefix = f"layer_{layer_number}"
            layers.append(
                {
                    "image": kwargs.get(f"{prefix}_image"),
                    "mask": kwargs.get(f"{prefix}_mask"),
                    "x": kwargs.get(f"{prefix}_x", 0),
                    "y": kwargs.get(f"{prefix}_y", 0),
                    "scale": kwargs.get(f"{prefix}_scale", 1.0),
                    "opacity": kwargs.get(f"{prefix}_opacity", 1.0),
                    "anchor": kwargs.get(f"{prefix}_anchor", "top_left"),
                    "fit_mode": kwargs.get(f"{prefix}_fit_mode", "none"),
                }
            )

        return (compose_layer_stack(background, layers),)
