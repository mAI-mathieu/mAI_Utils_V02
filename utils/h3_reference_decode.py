"""Bounded local media decoding for native ComfyUI H3 IMAGE/AUDIO sockets."""

import math
import sys

FPS = 24


def check_interrupted():
    management = sys.modules.get("comfy.model_management")
    callback = getattr(management, "throw_exception_if_processing_interrupted", None)
    if callback:
        callback()


def decode_settings(max_seconds, video_max_side):
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or not 0.25 <= max_seconds <= 15:
        raise ValueError("max_seconds must be between 0.25 and 15 seconds.")
    if isinstance(video_max_side, bool) or not isinstance(video_max_side, int) or not 64 <= video_max_side <= 2048:
        raise ValueError("video_max_side must be an integer between 64 and 2048.")


def fit_video_dimensions(width, height, max_side):
    if width < 1 or height < 1:
        raise ValueError("Reference video has invalid frame dimensions.")
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def load_reference_image(path):
    import numpy as np
    from PIL import Image, ImageOps
    import torch

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)


def load_reference_audio(path, max_seconds, *, optional=False):
    import av
    import numpy as np
    import torch

    sample_rate = 48000
    cap = round(max_seconds * sample_rate)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            if optional:
                return None
            raise ValueError("The selected audio file contains no audio stream.")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
        chunks = []
        samples = 0

        def append(frame):
            nonlocal samples
            values = frame.to_ndarray()[:, :cap - samples]
            if values.shape[1]:
                chunks.append(values.copy())
                samples += values.shape[1]

        # Align soundtrack timestamps to video time zero, including delayed audio.
        origin = None
        if optional and container.streams.video:
            video = container.streams.video[0]
            origin = float(video.start_time * video.time_base) if video.start_time is not None else 0.0
        audio_origin = None
        for frame in container.decode(stream):
            check_interrupted()
            for output in resampler.resample(frame):
                if audio_origin is None:
                    audio_origin = float(output.time) if output.time is not None else (origin or 0.0)
                    if origin is not None and audio_origin > origin:
                        delay = min(cap, round((audio_origin - origin) * sample_rate))
                        if delay:
                            chunks.append(np.zeros((2, delay), dtype=np.float32))
                            samples += delay
                # If the audio stream starts before the video, discard its lead-in.
                if origin is not None and output.time is not None and float(output.time) < origin:
                    skip = min(output.samples, max(0, round((origin - float(output.time)) * sample_rate)))
                    values = output.to_ndarray()[:, skip:skip + cap - samples]
                    if values.shape[1]:
                        chunks.append(values.copy())
                        samples += values.shape[1]
                else:
                    append(output)
                if samples >= cap:
                    break
            if samples >= cap:
                break
        if samples < cap:
            for output in resampler.resample(None):
                append(output)
        if not chunks:
            if optional:
                return None
            raise ValueError("The selected file contains no decodable audio samples.")
        waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def load_reference_video(path, max_seconds, video_max_side):
    import av
    import numpy as np
    import torch

    cap = math.floor(max_seconds * FPS)
    frames = []
    previous = None
    previous_time = 0.0
    end_time = 0.0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("The selected video file contains no video stream.")
        stream = container.streams.video[0]
        source_rate = float(stream.average_rate or FPS)
        if not math.isfinite(source_rate) or source_rate <= 0:
            source_rate = FPS
        origin = None
        dimensions = None
        for source_index, frame in enumerate(container.decode(stream)):
            check_interrupted()
            timestamp = float(frame.time) if frame.time is not None else source_index / source_rate
            if origin is None:
                origin = timestamp
                dimensions = fit_video_dimensions(frame.width, frame.height, video_max_side)
            timestamp = max(previous_time, timestamp - origin)
            # Sample at 24 fps using presentation timestamps, duplicating low-fps frames.
            while previous is not None and len(frames) < cap and len(frames) / FPS < timestamp - 1e-7:
                frames.append(previous)
            if len(frames) >= cap:
                break
            previous = frame.reformat(width=dimensions[0], height=dimensions[1], format="rgb24").to_ndarray()
            previous_time = timestamp
            duration = float(frame.duration * frame.time_base) if frame.duration and frame.time_base else 1 / source_rate
            end_time = timestamp + duration
        while previous is not None and len(frames) < cap and len(frames) / FPS < end_time - 1e-7:
            frames.append(previous)
    if len(frames) < 5:
        raise ValueError("H3 reference videos need at least 5 frames at 24 fps (about 0.21 seconds).")
    # Native H3 consumes 17*n+5 video frames; align soundtracks to this same end.
    count = len(frames) - (len(frames) - 5) % 17
    return torch.from_numpy(np.stack(frames[:count])).to(torch.float32).div_(255.0)


def load_reference_slots(references, entries, max_seconds=15.0, video_max_side=768, include_video_audio=True):
    """Unused sockets are None, which the native H3 optional inputs explicitly skip."""
    decode_settings(max_seconds, video_max_side)
    if not isinstance(include_video_audio, bool):
        raise ValueError("include_video_audio must be a boolean.")
    images, videos, video_audio, audios = [], [], [], []
    report = []
    for ref, entry in zip(references, entries):
        check_interrupted()
        try:
            if ref.kind == "image":
                images.append(load_reference_image(ref.path))
                socket = f"ref_image_{len(images) - 1}"
                detail = {"label": f"<Picture {len(images)}>"}
            elif ref.kind == "video":
                frames = load_reference_video(ref.path, max_seconds, video_max_side)
                videos.append(frames)
                soundtrack = load_reference_audio(ref.path, frames.shape[0] / FPS, optional=True) if include_video_audio else None
                video_audio.append(soundtrack)
                socket = f"ref_video_{len(videos) - 1}"
                detail = {"label": f"<Video {len(videos)}>", "frames": frames.shape[0], "fps": FPS,
                          "soundtrack_output": f"ref_video_audio_{len(videos) - 1}", "has_soundtrack": soundtrack is not None}
            else:
                audios.append(load_reference_audio(ref.path, max_seconds))
                socket = f"ref_audio_{len(audios) - 1}"
                detail = {}
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load reference '{entry['name']}': {exc}") from exc
        report.append({**entry, "output": socket, **detail})
    # Native H3 emits all video soundtrack labels before standalone audio labels.
    audio_index = 0
    for entry in report:
        if entry["kind"] == "video" and entry["has_soundtrack"]:
            audio_index += 1
            entry["soundtrack_label"] = f"<Audio {audio_index}>"
    for entry in report:
        if entry["kind"] == "audio":
            audio_index += 1
            entry["label"] = f"<Audio {audio_index}>"
    pad = lambda values, count: values + [None] * (count - len(values))
    return tuple(pad(images, 9) + pad(videos, 3) + pad(video_audio, 3) + pad(audios, 3)), report
