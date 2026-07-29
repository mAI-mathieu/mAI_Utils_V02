try:
    from ..utils.text_sequence import randomize_text_sequence
except ImportError:
    from utils.text_sequence import randomize_text_sequence


class MAITextSequenceRandomizer:
    CATEGORY = "mAI / Text"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "separator": (["comma", "line break"], {"default": "comma"}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
            }
        }

    def run(self, text, separator, seed):
        return (randomize_text_sequence(text, separator, seed),)
