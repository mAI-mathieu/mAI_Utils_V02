"""Pure parsing and interpolation for a LoRA strength timeline."""

from bisect import bisect_right
import math


DEFAULT_LORA_SCHEDULE = "0:1\n10:1\n40:0\n100:0"


def parse_lora_schedule(text):
    """Read ordered percent:strength points, separated by lines or commas."""
    points = []
    for entry in text.replace(",", "\n").splitlines():
        entry = entry.strip()
        if not entry:
            continue
        try:
            percent_text, strength_text = entry.split(":")
            percent = float(percent_text.strip().removesuffix("%"))
            strength = float(strength_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid LoRA schedule point {entry!r}. Use percent:strength, e.g. 25:0.5."
            ) from exc
        if not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError("LoRA schedule percentages must be finite numbers from 0 to 100.")
        if not math.isfinite(strength) or not -100 <= strength <= 100:
            raise ValueError("LoRA schedule strengths must be finite numbers from -100 to 100.")
        if points and percent <= points[-1][0]:
            raise ValueError("LoRA schedule percentages must be strictly increasing (no duplicates).")
        points.append((percent, strength))
    if not points:
        raise ValueError("LoRA schedule must contain at least one percent:strength point.")
    return tuple(points)


def schedule_strength(points, percent, interpolation="linear"):
    """Extend endpoint values and interpolate between neighboring points."""
    if interpolation not in ("linear", "constant"):
        raise ValueError(f"Unknown LoRA schedule interpolation: {interpolation}")
    if percent <= points[0][0]:
        return points[0][1]
    for (start, first), (end, last) in zip(points, points[1:]):
        if percent < end:
            if interpolation == "constant":
                return first
            return first + (last - first) * (percent - start) / (end - start)
    return points[-1][1]


def sigma_step_positions(sigmas):
    """Validate descending scheduler boundaries and prepare a binary search."""
    values = tuple(float(value) for value in sigmas)
    if len(values) < 2 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Scheduled LoRA needs at least two finite, nonnegative sampler sigmas.")
    if any(first <= last for first, last in zip(values, values[1:])):
        raise ValueError("Scheduled LoRA requires strictly decreasing sampler sigmas.")
    return tuple(-value for value in values)


def sigma_to_step_percent(sigma, positions):
    """Map sigma to actual step progress, including intermediate evaluations."""
    value = -float(sigma)
    if not math.isfinite(value):
        raise ValueError("Scheduled LoRA received a non-finite sampling sigma.")
    if value <= positions[0]:
        return 0.0
    if value >= positions[-1]:
        return 100.0
    index = bisect_right(positions, value) - 1
    fraction = (value - positions[index]) / (positions[index + 1] - positions[index])
    return 100.0 * (index + fraction) / (len(positions) - 1)
