try:
    from ..utils.h3_to_ltx_frame_adapter import select_ltx_frames
except ImportError:
    from utils.h3_to_ltx_frame_adapter import select_ltx_frames


class MAIH3ToLTXFrameAdapter:
    CATEGORY = "mAI / Image"
    FUNCTION = "adapt_frames"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "source_frames", "target_frames", "removed_frames")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    def adapt_frames(self, images):
        import torch

        # ComfyUI wraps a normal IMAGE batch (or a Python list IMAGE value).
        if isinstance(images, list) and len(images) == 1:
            if isinstance(images[0], list) or (
                isinstance(images[0], torch.Tensor) and images[0].ndim == 4
            ):
                images = images[0]
        selected = select_ltx_frames(images)
        source_frames = len(images)
        target_frames = len(selected)
        removed_frames = source_frames - target_frames
        if isinstance(selected, list):
            selected = torch.cat(
                [frame.unsqueeze(0) if frame.ndim == 3 else frame for frame in selected],
                dim=0,
            )

        prefix = "[mAI H3 to LTX Frame Adapter]"
        if removed_frames:
            print(f"{prefix} {source_frames} → {target_frames} frames, removed {removed_frames}")
        else:
            print(f"{prefix} {source_frames} frames already LTX-compatible")
        return selected, source_frames, target_frames, removed_frames
