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
