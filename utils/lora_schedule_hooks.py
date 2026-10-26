"""Adapter for ComfyUI's native weight hooks; imported only during execution."""

from comfy.hooks import HookKeyframe, HookKeyframeGroup

from .lora_schedule import schedule_strength, sigma_step_positions, sigma_to_step_percent


class LoRAScheduleKeyframes(HookKeyframeGroup):
    """Evaluate an arbitrary strength curve against the active sampler sigmas."""

    def __init__(self, points, interpolation):
        super().__init__()
        self.points = points
        self.interpolation = interpolation
        for percent, strength in points:
            self.add(HookKeyframe(strength, start_percent=percent / 100, guarantee_steps=0))
        self.reset()

    @property
    def strength(self):
        return self._schedule_strength

    def clone(self):
        return type(self)(self.points, self.interpolation)

    def reset(self):
        super().reset()
        self._schedule_strength = self.points[0][1]
        self._sample_sigmas = None
        self._positions = None

    def initialize_timesteps(self, model):
        # Native percentages describe noise time. This schedule uses actual steps
        # and gets its boundaries from sample_sigmas on the first evaluation.
        self.reset()

    def prepare_current_keyframe(self, curr_t, transformer_options):
        sigmas = transformer_options.get("sample_sigmas")
        if sigmas is None:
            raise ValueError("Scheduled LoRA requires a sampler that supplies sample_sigmas.")
        if sigmas is not self._sample_sigmas:
            values = sigmas.detach().cpu().tolist()
            self._positions = sigma_step_positions(values)
            self._sample_sigmas = sigmas
        percent = sigma_to_step_percent(float(curr_t), self._positions)
        strength = schedule_strength(self.points, percent, self.interpolation)
        changed = strength != self._schedule_strength
        self._schedule_strength = strength
        return changed
