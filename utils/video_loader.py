from itertools import islice


class _VideoPrefixDemux:
    """Stop sending video packets to the native decoder once the prefix is ready."""

    def __init__(self, container, frame_limit):
        self._container = container
        self.frame_limit = frame_limit
        self.frame_count = 0

    def __getattr__(self, name):
        return getattr(self._container, name)

    def demux(self, *streams):
        for packet in self._container.demux(*streams):
            if self.frame_count < self.frame_limit or packet.stream.type == "audio":
                yield packet


def load_video_components(video_type, path, frame_limit):
    """Keep ComfyUI's color, rotation, frame storage and audio handling."""
    if frame_limit == -1 or not callable(getattr(video_type, "_decode_frames", None)):
        # Older ComfyUI releases do not expose a streaming decode hook.
        return video_type(path).get_components()

    class LimitedVideo(video_type):
        def _decode_frames(self, container, video_stream, audio_stream, audio_frames, alphas):
            decode = super()._decode_frames
            if audio_stream is None:
                yield from islice(decode(container, video_stream, None, audio_frames, alphas), frame_limit)
                return

            demux = _VideoPrefixDemux(container, frame_limit)
            for image in decode(demux, video_stream, audio_stream, audio_frames, alphas):
                # A video packet can yield multiple images, notably on decoder flush.
                if demux.frame_count < frame_limit:
                    demux.frame_count += 1
                    yield image

    return LimitedVideo(path).get_components()


def get_frame_batch_info(frames):
    """Return frame count and dimensions for a ComfyUI IMAGE batch."""
    shape = getattr(frames, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError(
            "Decoded video frames must have shape "
            "[frames, height, width, channels]."
        )

    frame_count, height, width, channels = (int(value) for value in shape)
    if frame_count < 1:
        raise ValueError("The selected video contains no decodable frames.")
    if width < 1 or height < 1 or channels < 1:
        raise ValueError("The selected video has invalid frame dimensions.")

    return frame_count, width, height
