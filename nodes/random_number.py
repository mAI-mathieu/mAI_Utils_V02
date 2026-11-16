import math
import random
from sys import float_info


_FLOAT_STEPS = 2**53


def _sample_float(minimum, maximum):
    # Sample a closed grid so either endpoint is possible.
    fraction = random.randrange(_FLOAT_STEPS + 1) / _FLOAT_STEPS
    if fraction == 1:
        return maximum
    if minimum < 0 < maximum:
        # A difference across zero can overflow for finite bounds.
        return (1 - fraction) * minimum + fraction * maximum
    return min(maximum, minimum + fraction * (maximum - minimum))


class MAIRandomNumber:
    CATEGORY = "mAI / Utils"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int", "float")
    DESCRIPTION = "Draw random integer and float values from the same inclusive range."

    @classmethod
    def INPUT_TYPES(cls):
        bounds = {
            "min": -float_info.max,
            "max": float_info.max,
            "step": 0.01,
            "round": False,
        }
        return {
            "required": {
                "minimum": (
                    "FLOAT",
                    {**bounds, "default": 0.0, "tooltip": "Inclusive lower bound."},
                ),
                "maximum": (
                    "FLOAT",
                    {**bounds, "default": 1.0, "tooltip": "Inclusive upper bound."},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # ComfyUI treats NaN as a fresh cache fingerprint on every queue.
        return float("nan")

    def run(self, minimum, maximum):
        minimum, maximum = float(minimum), float(maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("minimum and maximum must be finite numbers")
        if minimum > maximum:
            raise ValueError("minimum must be less than or equal to maximum")

        first_integer = math.ceil(minimum)
        last_integer = math.floor(maximum)
        if first_integer <= last_integer:
            integer_value = random.randint(first_integer, last_integer)
        else:
            from comfy_execution.graph_utils import ExecutionBlocker

            integer_value = ExecutionBlocker(
                "The range contains no whole number. Use the float output or widen the range."
            )

        return (integer_value, _sample_float(minimum, maximum))
