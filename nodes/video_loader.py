import os

try:
    from ..utils.video_loader import get_frame_batch_info
except ImportError:
    from utils.video_loader import get_frame_batch_info


class MAIVideoLoader:
    CATEGORY = "mAI / IO"
    FUNCTION = "load_video"
    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO", "INT", "INT", "INT")
    RETURN_NAMES = ("frames", "fps", "audio", "frame_count", "width", "height")

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        files = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        videos = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "video": (sorted(videos), {"video_upload": True}),
            },
            "optional": {
                "frame_limit": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "tooltip": "Keep the first N frames. Use -1 for all frames. Zero is not valid.",
                    },
                ),
            },
        }

    def load_video(self, video, frame_limit=-1):
        if frame_limit < -1 or frame_limit == 0:
            raise ValueError("frame_limit must be -1 (all frames) or a positive integer.")

        import folder_paths
        from comfy_api.latest import InputImpl

        video_path = folder_paths.get_annotated_filepath(video)
        components = InputImpl.VideoFromFile(video_path).get_components()
        frames = components.images
        frame_count, width, height = get_frame_batch_info(frames)
        if 0 < frame_limit < frame_count:
            frames = frames[:frame_limit]
            frame_count = frame_limit

        return (
            frames,
            float(components.frame_rate),
            components.audio,
            frame_count,
            width,
            height,
        )

    @classmethod
    def IS_CHANGED(cls, video, frame_limit=-1):
        import folder_paths

        video_path = folder_paths.get_annotated_filepath(video)
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        import folder_paths

        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True
