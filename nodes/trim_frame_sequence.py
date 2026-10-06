try:
    from ..utils.frame_sequence import trim_frame_sequence_ends
except ImportError:
    from utils.frame_sequence import trim_frame_sequence_ends


class MAITrimFrameSequence:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "trim_start": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "tooltip": "Frames to remove from the beginning. Zero keeps this end.",
                    },
                ),
                "trim_end": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "tooltip": "Frames to remove from the end. Zero keeps this end.",
                    },
                ),
            }
        }

    def run(self, frames, trim_start=0, trim_end=0):
        # INPUT_IS_LIST also wraps scalar widgets; require one setting per sequence.
        settings = []
        for name, value in (("trim_start", trim_start), ("trim_end", trim_end)):
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise ValueError(f"{name} must contain one value for the whole sequence.")
                value = value[0]
            settings.append(value)
        # Also accept nodes returning a Python list as a single IMAGE value.
        if isinstance(frames, (list, tuple)) and len(frames) == 1:
            if isinstance(frames[0], (list, tuple)):
                frames = frames[0]
        result = trim_frame_sequence_ends(frames, *settings)
        return result, result.shape[0]
