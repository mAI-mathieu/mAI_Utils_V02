"""Select a dynamically typed value from JSON text."""

if __package__ == "nodes":
    from utils.json_parser import select_json_value
else:
    from ..utils.json_parser import select_json_value


class MAIJsonParser:
    CATEGORY = "mAI / Text"
    FUNCTION = "run"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    DESCRIPTION = (
        "Select a JSON value using a key or path such as items[0].name. "
        "Auto preserves strings, integers, floats and booleans. Objects, arrays "
        "and null become JSON text. Match the selected value to the receiving input."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {
                    "multiline": True,
                    "default": '{"prompt": "a cinematic landscape", "width": 1024}',
                    "dynamicPrompts": False,
                    "tooltip": "Paste valid JSON, or connect a STRING input.",
                }),
                "key_path": ("STRING", {
                    "default": "prompt",
                    "dynamicPrompts": False,
                    "tooltip": 'Examples: prompt, settings.width, items[0].name, ["key.with.dots"]. Blank or $ selects the root.',
                }),
                "output_type": (["auto", "string"], {
                    "default": "auto",
                    "tooltip": "Auto preserves scalar types. String always returns text. Objects, arrays and null always return JSON text.",
                }),
            },
        }

    def run(self, json_text, key_path="prompt", output_type="auto"):
        return (select_json_value(json_text, key_path, output_type),)
