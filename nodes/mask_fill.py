"""Fill a masked image region with a color or classical inpainting."""

try:
    from ..utils.mask_fill import fill_masked_image
except ImportError:
    from utils.mask_fill import fill_masked_image


class MAIMaskFill:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "method": (["color", "telea", "navier_stokes"], {"default": "color"}),
                "color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "Fill color in #RRGGBB format; used only by color mode.",
                    },
                ),
                "inpaint_radius": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 128.0,
                        "step": 0.5,
                        "tooltip": "Neighborhood radius in pixels for Telea and Navier-Stokes.",
                    },
                ),
            },
        }

    def run(self, image, mask, method="color", color="#000000", inpaint_radius=3.0):
        return (fill_masked_image(image, mask, method, color, inpaint_radius),)
