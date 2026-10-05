"""Chunked tensor-to-FFmpeg bridge for NVENC and explicit software encoding.

Only fixed FFmpeg argument lists are executed, with shell=False. No image
files, shell scripts, VHS imports or per-frame conversion loops are used.
"""

from dataclasses import dataclass, replace
from contextlib import ExitStack, suppress
import json
import logging
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import tempfile
import time

import torch
import torch.nn.functional as F

from .gpu_memory import gpu_memory_scope, video_memory_requirements


NVENC_CODECS = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
SOFTWARE_CODECS = ("libx264", "libx265", "libsvtav1")
CODECS = NVENC_CODECS + SOFTWARE_CODECS
PRESETS = tuple(f"p{i}" for i in range(1, 8))
SOFTWARE_PRESETS = ("ultrafast", "superfast", "veryfast", "medium", "slow", "slower", "veryslow")
SVT_AV1_PRESETS = (12, 10, 8, 6, 5, 4, 3)


class NVENCOutOfMemoryError(RuntimeError):
    """FFmpeg explicitly reported insufficient CUDA memory."""


@dataclass(frozen=True)
class VideoEncodeOptions:
    frame_rate: float = 24.0
    codec: str = "h264_nvenc"
    quality: int = 23
    preset: str = "p4"
    gpu_device: int = -1
    chunk_size: int = 0
    pingpong: bool = False
    loop_count: int = 0
    trim_to_audio: bool = False


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")


def validate_options(options):
    fps = options.frame_rate
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or not 0 < fps <= 1000:
        raise ValueError("frame_rate must be a finite number greater than 0 and at most 1000.")
    if options.codec not in CODECS:
        raise ValueError(f"Unknown video codec: {options.codec}")
    if options.preset not in PRESETS:
        raise ValueError(f"Unknown video preset: {options.preset}")
    for name, low, high in (("quality", 0, 51), ("gpu_device", -1, 128),
                            ("chunk_size", 0, 4096), ("loop_count", 0, 100)):
        _integer(getattr(options, name), name, low, high)
    for name in ("pingpong", "trim_to_audio"):
        if not isinstance(getattr(options, name), bool):
            raise ValueError(f"{name} must be a boolean.")


def validate_filename_prefix(prefix):
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("filename_prefix must be a nonempty relative filename prefix.")
    normalized = prefix.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or PureWindowsPath(prefix).drive or ".." in path.parts:
        raise ValueError("filename_prefix must stay inside ComfyUI's output/temp directory.")
    if normalized.endswith("/") or path.name in ("", "."):
        raise ValueError("filename_prefix must include a filename, not just a directory.")
    if any(ord(char) < 32 or char in '<>:"|?*' for char in normalized):
        raise ValueError("filename_prefix contains invalid filename characters.")
    return normalized


def get_video_dimensions(frames):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or any(d < 1 for d in frames.shape):
        raise ValueError("frames must be a nonempty IMAGE tensor [frames, height, width, channels].")
    if frames.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("frames must contain floating-point IMAGE pixels.")
    if frames.shape[3] not in (1, 3, 4):
        raise ValueError("frames must have 1, 3 or 4 channels; RGBA alpha is discarded.")
    n, h, w, _ = frames.shape
    # 4:2:0 encoding needs even dimensions. Preserve all original pixels by
    # adding at most one black column/row on the right/bottom, without resizing.
    return n, w + w % 2, h + h % 2


def validate_audio(audio):
    if audio is None:
        return None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio must be a ComfyUI AUDIO dictionary with waveform and sample_rate.")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("Audio waveform must have shape [1, channels, samples]; audio batches are not supported.")
    if not waveform.is_floating_point() or not 1 <= waveform.shape[1] <= 8 or waveform.shape[2] < 1:
        raise ValueError("Audio must contain floating-point samples and 1–8 nonempty channels.")
    _integer(audio["sample_rate"], "sample_rate", 1, 384000)
    return waveform, int(audio["sample_rate"])


def playback_frame_count(source_frames, pingpong=False, loop_count=0):
    cycle = 2 * source_frames - 2 if pingpong and source_frames > 1 else source_frames
    return cycle * (loop_count + 1)


def playback_indices(start, end, source_frames, pingpong, device="cpu"):
    cycle = playback_frame_count(source_frames, pingpong)
    offsets = torch.arange(start, end, device=device).remainder_(cycle)
    if pingpong and source_frames > 1:
        return torch.where(offsets < source_frames, offsets, cycle - offsets)
    return offsets


def calculate_video_timing(source_frames, options, audio_info=None):
    count = playback_frame_count(source_frames, options.pingpong, options.loop_count)
    if options.trim_to_audio and audio_info is not None:
        waveform, sample_rate = audio_info
        # Keep the frame containing the audio endpoint. The final fractional
        # frame interval is filled with silence so duration is frame aligned.
        audio_frames = max(1, math.ceil(waveform.shape[2] * options.frame_rate / sample_rate))
        count = min(count, audio_frames)
    return count, count / options.frame_rate


def find_ffmpeg():
    # Explicit overrides also work without imageio-ffmpeg installed. These are
    # executable paths/names, never shell commands or extra FFmpeg arguments.
    for variable in ("MAI_FFMPEG_EXE", "IMAGEIO_FFMPEG_EXE"):
        configured = os.environ.get(variable)
        if configured:
            executable = shutil.which(configured)
            if executable:
                return executable
            raise RuntimeError(f"{variable} does not identify an executable FFmpeg file. "
                               "Set it to the executable path, without quotes or arguments.")
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ModuleNotFoundError as exc:
        if exc.name != "imageio_ffmpeg":
            raise
    else:
        try:
            candidate = get_ffmpeg_exe()
        except (RuntimeError, OSError) as exc:
            raise RuntimeError(_missing_ffmpeg_message()) from exc
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError(_missing_ffmpeg_message())


def _missing_ffmpeg_message():
    return ("FFmpeg was not found. Install this pack's requirements in the Python environment "
            "running ComfyUI (python -m pip install -r requirements.txt), or install system FFmpeg. "
            "For Debian/Ubuntu Runpod images: apt-get update && apt-get install -y ffmpeg. "
            "For Modal: add .apt_install('ffmpeg') to the ComfyUI image and rebuild it. "
            "You can also set MAI_FFMPEG_EXE or IMAGEIO_FFMPEG_EXE to an existing executable. "
            "NVENC additionally requires a supported GPU and exposed NVIDIA video driver libraries; "
            "B200/B300 have no NVENC, so select libx264, libx265 or libsvtav1.")


def _process_flags():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def check_nvenc_support(executable, codec):
    info = subprocess.run([executable, "-hide_banner", "-h", f"encoder={codec}"],
                          capture_output=True, text=True, timeout=15, shell=False, **_process_flags())
    description = info.stdout + info.stderr
    if info.returncode or f"Encoder {codec} " not in description or "nv12" not in description:
        raise RuntimeError(f"This FFmpeg build does not support {codec} with NV12 input. "
                           "Use an NVENC-enabled build; no CPU fallback is performed. "
                           "B200/B300 have no NVENC hardware; select an explicit software codec instead.")


def check_software_support(executable, codec):
    info = subprocess.run([executable, "-hide_banner", "-h", f"encoder={codec}"],
                          capture_output=True, text=True, timeout=15, shell=False, **_process_flags())
    description = info.stdout + info.stderr
    if info.returncode or f"Encoder {codec} " not in description or "yuv420p" not in description:
        raise RuntimeError(f"This FFmpeg build does not support {codec} with YUV420P output. "
                           "Install FFmpeg with that software encoder, or select libx264 for H.264.")


def build_encode_command(executable, output_path, width, height, frame_count,
                         options, audio_path=None, audio_info=None, metadata_path=None):
    args = [str(executable), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pixel_format", "nv12", "-video_size", f"{width}x{height}",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-chroma_sample_location", "center",
            "-framerate", format(options.frame_rate, ".12g"), "-i", "pipe:0"]
    if audio_path is not None:
        waveform, sample_rate = audio_info
        args += ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(waveform.shape[1]), "-i", str(audio_path)]
    if metadata_path is not None:
        args += ["-f", "ffmetadata", "-i", str(metadata_path)]
    args += ["-map", "0:v:0"]
    if audio_path is not None:
        args += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-af", "apad"]
    else:
        args += ["-an"]
    args += ["-map_metadata", str(2 if audio_path is not None else 1) if metadata_path is not None else "-1",
             "-c:v", options.codec]
    if options.codec in NVENC_CODECS:
        args += ["-preset", options.preset, "-tune", "hq", "-rc", "vbr",
                 "-cq", str(options.quality), "-b:v", "0", "-gpu", str(options.gpu_device),
                 "-pix_fmt", "nv12"]
    else:
        preset_index = PRESETS.index(options.preset)
        preset = (str(SVT_AV1_PRESETS[preset_index]) if options.codec == "libsvtav1"
                  else SOFTWARE_PRESETS[preset_index])
        args += ["-preset", preset, "-crf", str(options.quality), "-pix_fmt", "yuv420p"]
    args += ["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-chroma_sample_location", "center",
             "-frames:v", str(frame_count), "-t", format(frame_count / options.frame_rate, ".12g"),
             "-movflags", "+faststart+use_metadata_tags", "-f", "mp4", str(output_path)]
    # NV12 has already been converted on the tensor's device using BT.709.
    # Software codecs deinterleave NV12 to planar YUV420P in FFmpeg; no second
    # RGB color conversion or unnecessary hardware-decode flag is inserted.
    return args


def _metadata_text(metadata):
    def escape(value):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for char in ("\\", "=", ";", "#", "\n"):
            text = text.replace(char, "\\" + char)
        return text
    lines = [";FFMETADATA1"]
    # Only the two workflow fields are accepted; no arbitrary command options.
    for key in ("prompt", "workflow"):
        if key in metadata:
            lines.append(f"{key}={escape(metadata[key])}")
    return "\n".join(lines) + "\n"


def _write_all(stream, data):
    view = memoryview(data).cast("B")
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise BrokenPipeError("FFmpeg stopped accepting input.")
        view = view[written:]


def _write_audio(path, audio_info, duration, check_interrupt):
    waveform, sample_rate = audio_info
    # Discard audio beyond the requested video; write interleaved PCM in bounded
    # sample chunks, not a second full waveform. AAC/muxing remain CPU tasks.
    samples = min(waveform.shape[2], math.ceil(duration * sample_rate))
    with path.open("wb") as stream:
        for start in range(0, samples, 65536):
            if check_interrupt:
                check_interrupt()
            chunk = waveform[0, :, start:start + min(65536, samples-start)].to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(chunk).all():
                raise ValueError("Audio contains non-finite samples.")
            chunk = chunk.clamp(-1, 1).transpose(0, 1).contiguous()
            _write_all(stream, chunk.numpy())  # Zero-copy CPU byte view, not image processing.


def _pack_nv12(frames, width, height):
    if not torch.isfinite(frames).all():
        raise ValueError("Frames contain non-finite pixels.")
    rgb = frames[..., :3] if frames.shape[3] != 1 else frames.expand(-1, -1, -1, 3)
    rgb = rgb.to(dtype=torch.float32).clamp(0, 1)
    if (frames.shape[1], frames.shape[2]) != (height, width):
        rgb = F.pad(rgb.permute(0, 3, 1, 2),
                    (0, width - frames.shape[2], 0, height - frames.shape[1])).permute(0, 2, 3, 1)
    # SDR encoded RGB -> limited-range BT.709 Y'CbCr. Chroma is box-filtered
    # across each 2x2 block for centered 4:2:0 sampling. All tensor work stays
    # on the original device; FP32 prevents half/BF16 coefficient error.
    y = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    y = (y * 219 + 16).round().clamp_(16, 235).to(torch.uint8)
    small = F.avg_pool2d(rgb.permute(0, 3, 1, 2), 2, 2)
    del rgb
    r, g, b = small.unbind(dim=1)
    luminance = r * 0.2126 + g * 0.7152 + b * 0.0722
    cb = ((b - luminance) * (224 / (2 * (1 - 0.0722))) + 128).round().clamp_(16, 240).to(torch.uint8)
    cr = ((r - luminance) * (224 / (2 * (1 - 0.2126))) + 128).round().clamp_(16, 240).to(torch.uint8)
    packed = torch.empty((len(frames), height * width * 3 // 2), device=frames.device, dtype=torch.uint8)
    packed[:, :height * width] = y.flatten(1)
    packed[:, height * width:] = torch.stack((cb, cr), dim=-1).flatten(1)
    return packed.to(device="cpu")


@torch.inference_mode()
def encode_video(frames, output_path, options=None, audio=None, metadata=None,
                 check_interrupt=None, on_progress=None):
    """Prepare CUDA headroom, encode, release scratch; retry NVENC OOM once."""
    options = options or VideoEncodeOptions()
    validate_options(options)
    source_count, width, height = get_video_dimensions(frames)
    validate_audio(audio)
    if check_interrupt:
        check_interrupt()
    chunk_size = options.chunk_size or max(1, min(32, (64 * 1024 * 1024) // (width * height * 3 // 2)))
    # FP32 RGB, conversion arithmetic, NV12, padding and pingpong selection.
    conversion_bytes = min(source_count, chunk_size) * width * height * 48
    if options.codec in NVENC_CODECS:
        budgets = video_memory_requirements(frames.device, options.gpu_device, conversion_bytes)
    else:
        # Software compression needs no external CUDA context or NVENC budget.
        budgets = {frames.device: conversion_bytes} if frames.device.type == "cuda" else {}
    for attempt in range(2):
        try:
            with ExitStack() as stack:
                for device, required in budgets.items():
                    stack.enter_context(gpu_memory_scope(device, required, force_offload=bool(attempt)))
                return _encode_video(frames, output_path, options, audio, metadata,
                                     check_interrupt, on_progress)
        except NVENCOutOfMemoryError:
            if attempt:
                raise
            if check_interrupt:
                check_interrupt()
            logging.warning("mAI GPU Video Combine: NVENC ran out of CUDA memory; retrying once "
                            "after requesting ComfyUI model offloading, with one-frame conversion chunks.")
            options = replace(options, chunk_size=1)


def _encode_video(frames, output_path, options=None, audio=None, metadata=None,
                  check_interrupt=None, on_progress=None):
    """Encode one MP4; return encoded frame count and duration.

    The fixed FFmpeg child consumes chunked NV12 through stdin. Temporary PCM
    and metadata are cleaned on every exit. Only a complete video is published;
    an existing destination is never overwritten, including on failure.
    """
    options = options or VideoEncodeOptions()
    validate_options(options)
    source_count, width, height = get_video_dimensions(frames)
    audio_info = validate_audio(audio)
    count, duration = calculate_video_timing(source_count, options, audio_info)
    output = Path(output_path)
    if output.suffix.lower() != ".mp4":
        raise ValueError("GPU Video Combine currently writes MP4 files only.")
    if output.exists():
        raise FileExistsError(f"Video destination already exists: {output.name}")
    executable = find_ffmpeg()
    hardware = options.codec in NVENC_CODECS
    if hardware:
        check_nvenc_support(executable, options.codec)
    else:
        check_software_support(executable, options.codec)
    if check_interrupt:
        check_interrupt()
    output.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = options.chunk_size or max(1, min(32, (64 * 1024 * 1024) // (width * height * 3 // 2)))
    with tempfile.TemporaryDirectory(prefix=".mai_video_", dir=output.parent) as temp:
        temp = Path(temp)
        audio_path = temp / "audio.f32" if audio_info is not None else None
        if audio_path is not None:
            _write_audio(audio_path, audio_info, duration, check_interrupt)
        metadata_path = temp / "workflow.ffmeta" if metadata else None
        if metadata_path is not None:
            metadata_path.write_text(_metadata_text(metadata), encoding="utf-8")
        encoded = temp / "video.mp4"
        command = build_encode_command(executable, encoded, width, height, count, options,
                                       audio_path, audio_info, metadata_path)
        with tempfile.TemporaryFile(mode="w+b", dir=temp) as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=stderr, shell=False, **_process_flags())
            pipe_error = None
            try:
                try:
                    for start in range(0, count, chunk_size):
                        if check_interrupt:
                            check_interrupt()
                        end = min(count, start + chunk_size)
                        if not options.pingpong and options.loop_count == 0:
                            chunk = frames[start:end]
                        else:
                            indices = playback_indices(start, end, source_count, options.pingpong, frames.device)
                            chunk = frames.index_select(0, indices)
                        packed = _pack_nv12(chunk, width, height)
                        _write_all(process.stdin, packed.numpy())
                        del packed, chunk
                        if on_progress:
                            on_progress(end, count)
                    process.stdin.close()
                except InterruptedError:
                    raise
                except (BrokenPipeError, OSError) as exc:
                    pipe_error = exc
                    with suppress(BrokenPipeError, OSError):
                        process.stdin.close()
                deadline = time.monotonic() + 120
                while process.poll() is None:
                    if check_interrupt:
                        check_interrupt()
                    if time.monotonic() >= deadline:
                        raise RuntimeError("FFmpeg did not finish within 120 seconds after the last frame.")
                    try:
                        process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode != 0 or pipe_error is not None:
                    stderr.seek(0)
                    details = stderr.read()[-8000:].decode("utf-8", errors="replace").strip()
                    error_type = NVENCOutOfMemoryError if hardware and "CUDA_ERROR_OUT_OF_MEMORY" in details else RuntimeError
                    hint = ("Check the NVIDIA driver, codec support, frame dimensions and available GPU memory. "
                            "Linux containers must expose libnvidia-encode.so.1 (NVIDIA video capability). "
                            "B200/B300 have no NVENC; select libx264, libx265 or libsvtav1. "
                            "No CPU fallback is performed." if hardware else
                            "Check FFmpeg software codec support, frame dimensions and available CPU memory.")
                    label = "NVENC" if hardware else "Software"
                    raise error_type(f"{label} video encoding failed ({options.codec}). {hint}\n"
                                     + (details or str(pipe_error))) from pipe_error
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if not process.stdin.closed:
                    with suppress(BrokenPipeError, OSError):
                        process.stdin.close()
        if not encoded.is_file() or encoded.stat().st_size == 0:
            raise RuntimeError("FFmpeg finished without producing a video.")
        # Exclusive reservation prevents concurrent runs from overwriting a
        # destination. Replace only the empty file owned by this execution.
        with output.open("xb"):
            pass
        try:
            os.replace(encoded, output)
        except OSError:
            output.unlink()
            raise
    return count, duration
