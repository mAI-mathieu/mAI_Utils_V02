try:
    from ..utils.type_conversion import OUTPUT_TYPES, convert_value
except ImportError:
    from utils.type_conversion import OUTPUT_TYPES, convert_value


class MAITypeConverterNode:
    CATEGORY = "mAI / Utils"
    # The frontend gives this single socket the selected concrete type.
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {"forceInput": True}),
                "output_type": (list(OUTPUT_TYPES), {"default": "string"}),
                "strict": ("BOOLEAN", {"default": False}),
            }
        }

    def convert(self, value, output_type="string", strict=False):
        return (convert_value(value, output_type, strict),)
