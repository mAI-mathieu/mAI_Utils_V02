try:
    from ..utils.pixel_drift import fix_pixel_drift
except ImportError:
    from utils.pixel_drift import fix_pixel_drift


class MAIPixelDriftFix:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "alignment_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "Image sent to the edit model; alignment reference only."}),
                "edited_image": ("IMAGE", {"tooltip": "Image to correct. Passed through exactly when alignment is unsafe."}),
                "min_similarity": ("FLOAT", {
                    "default": 0.60, "min": 0.30, "max": 1.0, "step": 0.05,
                    "tooltip": "Required fraction of matching structural tiles. Higher is more conservative; this is not an AI similarity probability.",
                }),
                "max_drift": ("FLOAT", {
                    "default": 0.15, "min": 0.01, "max": 0.30, "step": 0.01,
                    "tooltip": "Maximum corner movement as a fraction of image width/height. Rejects large reframing and extreme transforms.",
                }),
            },
            "optional": {
                "mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "Auto detects small shifts and chooses alignment checks per pair. Manual uses the two tuning controls.",
                }),
            },
        }

    def run(self, source_image, edited_image, min_similarity=0.60, max_drift=0.15,
            mode="manual"):
        # Calls from workflows saved before mode existed retain their behavior.
        return fix_pixel_drift(source_image, edited_image, min_similarity, max_drift, mode)
