import math
from random import Random
from sys import float_info


_FLOAT_STEPS = 2**53


def _sample_float(minimum, maximum, rng):
    # Sample a closed grid so either endpoint is possible.
    fraction = rng.randrange(_FLOAT_STEPS + 1) / _FLOAT_STEPS
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
            "optional": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**53 - 1,
                        "control_after_generate": True,
                        "tooltip": "Reuse a seed and range to repeat both draws.",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **inputs):
        # Older prompts can omit the seed and still need a fresh draw every queue.
        if "seed" not in inputs:
            return float("nan")
        return None

    def run(self, minimum, maximum, seed=None):
        minimum, maximum = float(minimum), float(maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("minimum and maximum must be finite numbers")
        if minimum > maximum:
            raise ValueError("minimum must be less than or equal to maximum")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer")

        rng = Random(seed)
        first_integer = math.ceil(minimum)
        last_integer = math.floor(maximum)
        if first_integer <= last_integer:
            integer_value = rng.randint(first_integer, last_integer)
        else:
            from comfy_execution.graph_utils import ExecutionBlocker

            integer_value = ExecutionBlocker(
                "The range contains no whole number. Use the float output or widen the range."
            )

        return (integer_value, _sample_float(minimum, maximum, rng))
