class MAIExampleTextNode:
    """Small example node that cleans a text string.

    Replace this class with your real node implementation.
    """

    CATEGORY = "mAI / Template"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello ComfyUI"}),
                "strip_whitespace": ("BOOLEAN", {"default": True}),
            }
        }

    def run(self, text, strip_whitespace):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        result = text.strip() if strip_whitespace else text
        return (result,)
