"""ComfyUI adapter for the independent tensor loop optimizer."""

import json
import sys

import torch

try:
    from ..utils.seamless_loop import LoopCandidate, LoopOptions, is_cuda_compatibility_error, optimize_loop
    from ..utils.seamless_loop_audio import render_loop_audio, validate_loop_audio
except ImportError:
    from utils.seamless_loop import LoopCandidate, LoopOptions, is_cuda_compatibility_error, optimize_loop
    from utils.seamless_loop_audio import render_loop_audio, validate_loop_audio


class MAIAutoSeamlessLoop:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "INT", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("images", "fps", "frame_count", "trim_start", "trim_end", "overlap", "report", "audio")
    DESCRIPTION = (
        "Jointly search trims and crossfades for an ordered video IMAGE batch. "
        "Score the start/end transition for microjumps, motion, brightness, contrast and ghosting. "
        "Optional audio follows the same trims, phase rotation and overlap. "
        "Output may be shorter; FPS is unchanged. No video encoder or model required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "quality": (["fast", "balanced", "high"], {"default": "balanced"}),
                "max_trim_start": ("INT", {"default": 8, "min": 0, "max": 10000}),
                "max_trim_end": ("INT", {"default": 8, "min": 0, "max": 10000}),
                "max_fade": ("INT", {"default": 12, "min": 0, "max": 10000}),
                "min_retained_percent": ("FLOAT", {"default": 70.0, "min": 0.1, "max": 100.0,
                                                   "tooltip": "Minimum FINAL cycle length, after trims and overlap, as % of input frames."}),
                "advanced_controls": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "blend_space": (["srgb", "linear"], {"default": "srgb"}),
                **{name + "_weight": ("FLOAT", {"default": value, "min": 0.0, "max": 10.0, "step": 0.01})
                   for name, value in (("appearance", 1.0), ("exposure", 0.5), ("motion", 0.6),
                                       ("smoothness", 0.4), ("ghosting", 1.0), ("duration", 0.25), ("fade", 0.12))},
                "satisfactory_score": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 10.0, "step": 0.001}),
                "tie_tolerance": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.001}),
                **{name: ("INT", {"default": -1, "min": -1, "max": 10000,
                                  "tooltip": "-1 searches automatically. Nonnegative values fix this dimension within the basic bounds."})
                   for name in ("manual_trim_start", "manual_trim_end", "manual_overlap")},
                "audio": ("AUDIO",),
            },
        }

    def run(self, images, fps=24.0, device="auto", quality="balanced", max_trim_start=8,
            max_trim_end=8, max_fade=12, min_retained_percent=70.0, advanced_controls=False,
            audio=None, **advanced):
        options = LoopOptions(fps=fps, device=device, quality=quality, max_trim_start=max_trim_start,
                              max_trim_end=max_trim_end, max_fade=max_fade,
                              min_retained_percent=min_retained_percent, **advanced)
        management = sys.modules.get("comfy.model_management")
        progress_class = getattr(sys.modules.get("comfy.utils"), "ProgressBar", None)
        interrupt = getattr(management, "throw_exception_if_processing_interrupted", None)
        preferred = None
        configured_error = None
        if interrupt:
            interrupt()
        validate_loop_audio(audio, check_interrupt=interrupt)
        if device == "auto" and management is not None:
            try:
                preferred = management.get_torch_device()
            except RuntimeError as exc:
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and not is_cuda_compatibility_error(exc):
                    raise
                preferred = torch.device("cpu")
                configured_error = f"ComfyUI CUDA device query failed: {exc}"
        output_device = management.intermediate_device() if management is not None else "cpu"
        progress = progress_class(100) if progress_class else None

        def report_progress(current, total):
            if progress:
                progress.update_absolute(current, total)

        result, report = optimize_loop(images, options, preferred_device=preferred, output_device=output_device,
                                       check_interrupt=interrupt, on_progress=report_progress)
        if configured_error:
            report["backend"]["fallback_reason"] = configured_error
        selected = report["selected"]
        loop_audio, report["audio"] = render_loop_audio(
            audio, len(images), float(fps), LoopCandidate(selected["trim_start"], selected["trim_end"],
                                                        selected["overlap"]), check_interrupt=interrupt)
        return (result, float(fps), len(result), selected["trim_start"], selected["trim_end"],
                selected["overlap"], json.dumps(report, allow_nan=False), loop_audio)
