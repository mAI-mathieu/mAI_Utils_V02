"""ComfyUI wrapper for automatic source-preserving edit compositing."""

import warnings

import torch

try:
    from ..utils.image_preserve_source import preserve_source_image
    from ..utils.masked_image_composite import BLEND_MODES
except ImportError:
    from utils.image_preserve_source import preserve_source_image
    from utils.masked_image_composite import BLEND_MODES


def _is_cuda_failure(error):
    """Retry hardware failures, while leaving validation/programming errors visible."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return any(marker in message for marker in (
        "cuda error", "cuda out of memory", "cuda driver", "cuda initialization",
        "cuda unavailable", "cuda is not available", "no kernel image",
        "invalid device function", "cublas_status", "cudnn_status", "cudnn error",
        "not compiled with cuda", "found no nvidia driver", "cuda-capable device",
    ))


def _cpu_input(value):
    return value.detach().to(device="cpu") if isinstance(value, torch.Tensor) else value


class MAIImagePreserveSource:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "change_mask")
    DESCRIPTION = (
        "Keep the original image except where a substantial edit is detected. "
        "Compare aligned images at pixel and blurred scales, then blend their "
        "join with feather, multiband, or content-aware seams. Select auto to "
        "estimate drift, preserve edit sharpness, and retain the edited image "
        "when the composition no longer matches. Matching areas can recover "
        "some source detail without restoring the old lighting or color."
        " CUDA memory or compatibility failures retry once on CPU."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "edited_image": ("IMAGE",),
                "difference_threshold": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Ignore RGB differences below this value. Raise to preserve more source.",
                }),
                "transition_width": ("FLOAT", {
                    "default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Smooth transition above the threshold; zero makes a hard selection.",
                }),
                "comparison_blur": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 32.0, "step": 0.1,
                    "tooltip": "Gaussian spread in pixels for broad-change detection. Zero compares raw pixels.",
                }),
                "detail_sensitivity": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 favors broad changes; 1 detects all raw pixel changes, including texture drift.",
                }),
                "mask_expand": ("INT", {"default": 2, "min": 0, "max": 128,
                    "tooltip": "Expand detected edits by this many pixels before feathering."}),
                "mask_blur": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 32.0, "step": 0.1,
                    "tooltip": "Gaussian mask feather spread in pixels. Zero disables feathering.",
                }),
                "blend_mode": (list(BLEND_MODES), {"default": "multiband",
                    "tooltip": "Content-aware modes optimize the seam inside the gray mask band."}),
                "multiband_levels": ("INT", {"default": 5, "min": 1, "max": 10,
                    "tooltip": "Pyramid levels in multiband modes. More levels blend broader transitions."}),
            },
            "optional": {
                "edit_mask": ("MASK", {
                    "tooltip": "Optional allowed edit area. Black always protects the original; white allows detection.",
                }),
                "mode": (["manual", "auto"], {"default": "manual",
                    "tooltip": "Auto estimates edit regions and blending settings per image; manual uses the tuning controls.",
                }),
            },
        }

    def run(
        self, original_image, edited_image, difference_threshold=0.08,
        transition_width=0.04, comparison_blur=2.0, detail_sensitivity=0.25,
        mask_expand=2, mask_blur=2.0, blend_mode="multiband", multiband_levels=5,
        edit_mask=None, mode="manual",
    ):
        settings = dict(
            difference_threshold=difference_threshold, transition_width=transition_width,
            comparison_blur=comparison_blur, detail_sensitivity=detail_sensitivity,
            mask_expand=mask_expand, mask_blur=mask_blur, blend_mode=blend_mode,
            multiband_levels=multiband_levels, mode=mode,
        )
        try:
            return preserve_source_image(original_image, edited_image, edit_mask=edit_mask, **settings)
        except RuntimeError as exc:
            cuda_input = any(
                isinstance(value, torch.Tensor) and value.device.type == "cuda"
                for value in (original_image, edited_image, edit_mask)
            )
            if not cuda_input or not _is_cuda_failure(exc):
                raise
            warnings.warn(
                f"mAI image preserve source: CUDA failed; retrying on CPU. Reason: {exc}",
                RuntimeWarning, stacklevel=2,
            )
        # Retry after leaving the handler so failed GPU work is not retained by
        # its traceback. Do not move the completed result back to a failing GPU.
        # CPU/transfer failures propagate; this is a single retry, not a loop.
        return preserve_source_image(
            _cpu_input(original_image), _cpu_input(edited_image),
            edit_mask=_cpu_input(edit_mask), **settings,
        )
