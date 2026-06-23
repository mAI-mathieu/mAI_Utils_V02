import random


class MAIRandomLine:
    CATEGORY = "mAI / Utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("line",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, text):
        return random.random()

    def run(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return ("",)

        return (random.choice(lines),)
