try:
    from ..utils.frame_sequence import append_loop_fade
except ImportError:
    from utils.frame_sequence import append_loop_fade


class MAIFrameLoopFade:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fade_frames": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Append this many frames fading the last image into the first. Zero disables the fade.",
                    },
                ),
            }
        }

    def run(self, images, fade_frames=24):
        # ComfyUI wraps scalar widgets when INPUT_IS_LIST is enabled.
        if isinstance(fade_frames, (list, tuple)):
            if len(fade_frames) != 1:
                raise ValueError("fade_frames must contain one value for the whole sequence.")
            fade_frames = fade_frames[0]
        # Also support a Python list returned as a single IMAGE value.
        if isinstance(images, (list, tuple)) and len(images) == 1:
            if isinstance(images[0], (list, tuple)):
                images = images[0]
        result = append_loop_fade(images, fade_frames)
        return result, result.shape[0]
