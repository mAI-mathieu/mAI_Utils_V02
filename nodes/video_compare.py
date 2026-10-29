"""Preview two IMAGE sequences with frame-synchronized frontend playback."""

from pathlib import Path
import shutil
import sys
import tempfile

try:
    from ..utils.video_compare import compare_batches, compare_settings, resolve_compare_device, write_compare_frames
except ImportError:
    from utils.video_compare import compare_batches, compare_settings, resolve_compare_device, write_compare_frames


class MAIVideoCompare:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    DESCRIPTION = (
        "Compare two frame sequences with synchronized play/pause, loop and hover reveal. "
        "Preview resizing and pixel conversion use ComfyUI's CUDA GPU when available. "
        "Frame zero is aligned; playback ends at the shorter sequence. No audio or video export."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames_a": ("IMAGE",),
            "frames_b": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 120.0, "step": 0.01}),
            "preview_max_size": ("INT", {
                "default": 768, "min": 64, "max": 4096, "step": 1,
                "tooltip": "Maximum preview side in pixels. Smaller previews load faster; source frames are unchanged.",
            }),
        }}

    def run(self, frames_a, frames_b, fps=24.0, preview_max_size=768):
        import folder_paths

        settings = []
        for name, value in (("fps", fps), ("preview_max_size", preview_max_size)):
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise ValueError(f"{name} must contain one value for the whole comparison.")
                value = value[0]
            settings.append(value)
        fps, preview_max_size = settings
        compare_settings(fps, preview_max_size)
        a = compare_batches(frames_a, "frames_a")
        b = compare_batches(frames_b, "frames_b")
        counts = [sum(batch.shape[0] for batch in batches) for batches in (a, b)]
        count = min(counts)
        compute_device = resolve_compare_device(a[0].device)
        base = Path(folder_paths.get_temp_directory())
        base.mkdir(parents=True, exist_ok=True)
        management = sys.modules.get("comfy.model_management")
        interrupt = getattr(management, "throw_exception_if_processing_interrupted", None)
        progress_class = getattr(sys.modules.get("comfy.utils"), "ProgressBar", None)
        progress = progress_class(count * 2) if progress_class else None

        # Unique folders keep cached executions and multiple comparison nodes independent.
        # Successful previews stay in ComfyUI temp for cached results until
        # ComfyUI clears it. No directory rename is needed (Windows image scanners
        # can briefly hold new PNGs open and block a rename of their parent).
        ready = Path(tempfile.mkdtemp(prefix="mai_compare_", dir=base))
        complete = False
        try:
            for prefix, batches in (("a", a), ("b", b)):
                write_compare_frames(batches, ready, prefix, count, preview_max_size,
                                     interrupt, (lambda: progress.update(1)) if progress else None,
                                     compute_device=compute_device)
            complete = True
        finally:
            if not complete:
                shutil.rmtree(ready)
        height, width = a[0].shape[1:3]
        return {"ui": {"video_compare": [{
            "subfolder": ready.name, "frame_count": count, "source_counts": counts,
            "fps": float(fps), "width": width, "height": height,
            "preview_device": str(compute_device),
        }]}, "result": ()}
