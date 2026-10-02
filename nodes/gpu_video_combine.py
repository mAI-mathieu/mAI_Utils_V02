"""ComfyUI output node for frames + optional AUDIO -> NVENC MP4."""

from pathlib import Path
import sys
import uuid

try:
    from ..utils.video_encoding import (
        CODECS, PRESETS, VideoEncodeOptions, encode_video, get_video_dimensions,
        validate_filename_prefix,
    )
except ImportError:
    from utils.video_encoding import (
        CODECS, PRESETS, VideoEncodeOptions, encode_video, get_video_dimensions,
        validate_filename_prefix,
    )


class MAIGPUVideoCombine:
    CATEGORY = "mAI / IO"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("VHS_FILENAMES", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("filenames", "file_path", "frame_count", "duration")
    DESCRIPTION = (
        "Combine IMAGE frames and optional AUDIO into MP4 using NVIDIA NVENC. "
        "RGB-to-NV12 conversion runs on the tensor's device; encoding uses NVENC. FFmpeg input crosses a CPU buffer. "
        "Odd dimensions are padded right/bottom to even sizes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "video/mAI"}),
                "codec": (list(CODECS), {"default": "h264_nvenc"}),
                "quality": ("INT", {"default": 23, "min": 0, "max": 51,
                                    "tooltip": "NVENC CQ: lower is higher quality and larger files. This is not software CRF."}),
                "preset": (list(PRESETS), {"default": "p4"}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100,
                                       "tooltip": "Additional encoded repeats. Audio plays once, then pads with silence."}),
                "trim_to_audio": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "gpu_device": ("INT", {"default": -1, "min": -1, "max": 128,
                                       "tooltip": "FFmpeg NVENC device index; -1 selects an available GPU."}),
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 4096,
                                       "tooltip": "Frames converted/transferred per chunk. 0 chooses up to 64 MiB of packed pixels."}),
            },
            "optional": {"audio": ("AUDIO",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def run(self, frames, frame_rate=24.0, filename_prefix="video/mAI", codec="h264_nvenc",
            quality=23, preset="p4", pingpong=False, loop_count=0, trim_to_audio=False,
            save_output=True, save_metadata=True, gpu_device=-1, chunk_size=0,
            audio=None, prompt=None, extra_pnginfo=None):
        import folder_paths

        prefix = validate_filename_prefix(filename_prefix)
        _, width, height = get_video_dimensions(frames)
        options = VideoEncodeOptions(frame_rate, codec, quality, preset, gpu_device,
                                     chunk_size, pingpong, loop_count, trim_to_audio)
        base = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, base, width, height)
        file = f"{filename}_{counter:05}_{uuid.uuid4().hex[:8]}.mp4"
        output = Path(folder) / file
        metadata = {}
        args = getattr(sys.modules.get("comfy.cli_args"), "args", None)
        if save_metadata and not getattr(args, "disable_metadata", False):
            if prompt is not None:
                metadata["prompt"] = prompt
            if extra_pnginfo and "workflow" in extra_pnginfo:
                metadata["workflow"] = extra_pnginfo["workflow"]

        # ComfyUI already loads these modules. Read runtime hooks lazily so
        # standalone imports/tests never initialize CUDA or require ComfyUI.
        management = sys.modules.get("comfy.model_management")
        interrupt = getattr(management, "throw_exception_if_processing_interrupted", None)
        progress_class = getattr(sys.modules.get("comfy.utils"), "ProgressBar", None)
        progress = None

        def report_progress(current, total):
            nonlocal progress
            if progress_class:
                if progress is None:
                    progress = progress_class(total)
                progress.update_absolute(current)

        count, duration = encode_video(frames, output, options, audio, metadata,
                                       check_interrupt=interrupt, on_progress=report_progress)
        path = str(output.resolve())
        preview = {"filename": file, "subfolder": subfolder, "type": "output" if save_output else "temp"}
        return {
            "ui": {"images": [preview], "animated": (True,)},
            "result": ((save_output, [path]), path, count, duration),
        }
