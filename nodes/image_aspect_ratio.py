try:
    from ..utils.aspect_ratios import closest_aspect_ratio
except ImportError:
    from utils.aspect_ratios import closest_aspect_ratio


class MAIImageAspectRatio:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("aspect_ratio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def run(self, image):
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) != 4 or any(size <= 0 for size in shape):
            raise ValueError(
                "Image must have non-empty shape [batch, height, width, channels]."
            )
        return (closest_aspect_ratio(int(shape[2]), int(shape[1])),)
