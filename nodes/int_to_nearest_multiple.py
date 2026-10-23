"""Round an integer up or down to a divisible value."""

if __package__ == "nodes":
    from utils.integer_multiples import round_to_multiple
else:
    from ..utils.integer_multiples import round_to_multiple


class MAIIntToNearestMultiple:
    CATEGORY = "mAI / Utils"
    FUNCTION = "run"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    DESCRIPTION = (
        "Round an integer up or down to the nearest multiple of a positive "
        "integer. Exact multiples stay unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": -(2**53 - 1),
                    "max": 2**53 - 1,
                    "step": 1,
                }),
                "multiple": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 2**53 - 1,
                    "step": 1,
                    "tooltip": "Positive integer that the output must be divisible by.",
                }),
                "rounding": (["round up", "round down"], {
                    "default": "round down",
                    "tooltip": "Up rounds toward positive infinity; down toward negative infinity.",
                }),
            },
        }

    def run(self, value, multiple, rounding="round down"):
        return (round_to_multiple(value, multiple, rounding),)
