"""Compare an image batch property for conditional workflow branching."""

if __package__ == "nodes":
    from utils.image_logic_check import compare_values, image_property_value
else:
    from ..utils.image_logic_check import compare_values, image_property_value


class MAIImageLogicCheck:
    CATEGORY = "mAI / Logic"
    FUNCTION = "run"
    RETURN_TYPES = ("BOOLEAN", "FLOAT")
    RETURN_NAMES = ("result", "actual_value")
    DESCRIPTION = (
        "Compare an IMAGE batch property and output a boolean for branching. "
        "Reads only tensor dimensions, without copying or moving image data."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "property": ([
                    "Megapixels", "Width", "Height", "Aspect Ratio", "Batch Size",
                ], {"default": "Megapixels"}),
                "operator": ([">", ">=", "<", "<=", "==", "!="], {"default": ">"}),
                "compare_value": ("FLOAT", {
                    "default": 2.2,
                    "min": 0.0,
                    "max": 1e15,
                    "step": 0.01,
                }),
            },
        }

    def run(self, image, property, operator, compare_value):
        actual_value = image_property_value(getattr(image, "shape", None), property)
        result = compare_values(actual_value, operator, compare_value)
        return (result, actual_value)
