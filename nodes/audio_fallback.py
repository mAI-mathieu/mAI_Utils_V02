"""Supply silence when an optional audio stream is missing."""

try:
    from ..utils.audio_fallback import resolve_audio_fallback
except ImportError:
    from utils.audio_fallback import resolve_audio_fallback


class MAIAudioFallback:
    CATEGORY = "mAI / Utils"
    FUNCTION = "run"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    DESCRIPTION = (
        "Pass through existing audio unchanged. Missing or empty audio becomes "
        "48 kHz stereo silence lasting frame_count / fps seconds."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {
                    "default": 120, "min": 1, "max": 1000000,
                    "tooltip": "Number of video frames covered by fallback silence.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                    "tooltip": "Video frame rate; used only when audio is missing or empty.",
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    def run(self, frame_count=120, fps=24.0, audio=None):
        return (resolve_audio_fallback(audio, frame_count, fps),)
