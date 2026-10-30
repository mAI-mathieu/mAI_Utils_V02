"""Hold an ordered set of image, audio and video references for MiniMax H3."""

import json

try:
    from ..utils.h3_reference_media import reference_fingerprint, resolve_references
    from ..utils.h3_reference_decode import load_reference_slots
except ImportError:
    from utils.h3_reference_media import reference_fingerprint, resolve_references
    from utils.h3_reference_decode import load_reference_slots


class MAIH3MediaReferences:
    CATEGORY = "mAI / Input"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",) * 12 + ("AUDIO",) * 6 + ("STRING",)
    RETURN_NAMES = (tuple(f"ref_image_{index}" for index in range(9))
                    + tuple(f"ref_video_{index}" for index in range(3))
                    + tuple(f"ref_video_audio_{index}" for index in range(3))
                    + tuple(f"ref_audio_{index}" for index in range(3)) + ("manifest",))
    DESCRIPTION = (
        "Drop images, audio or videos, or choose a folder. Preview and drag cards to set H3 reference order. "
        "Connect matching numbered outputs to MiniMax H3 Reference to Video. Unused slots return None."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"media_manifest": ("STRING", {
            "default": "[]", "multiline": True,
            "tooltip": "Ordered uploaded media, managed by the preview panel. Empty by default.",
        }), "max_seconds": ("FLOAT", {"default": 15.0, "min": 0.25, "max": 15.0, "step": 0.25,
            "tooltip": "Load the beginning of audio/video up to this duration. Videos are sampled at 24 fps."}),
            "video_max_side": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 1,
                "tooltip": "Maximum video frame side, preserving aspect ratio without upscaling. Smaller values reduce RAM."}),
            "include_video_audio": ("BOOLEAN", {"default": True,
                "tooltip": "Extract each video's soundtrack to its matching ref_video_audio output."})}}

    @classmethod
    def IS_CHANGED(cls, media_manifest="[]", **kwargs):
        return reference_fingerprint(media_manifest)

    def run(self, media_manifest="[]", max_seconds=15.0, video_max_side=768, include_video_audio=True):
        references, entries = resolve_references(media_manifest)
        slots, report = load_reference_slots(references, entries, max_seconds, video_max_side, include_video_audio)
        return (*slots, json.dumps(report, ensure_ascii=False, indent=2))
