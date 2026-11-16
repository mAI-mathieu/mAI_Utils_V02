"""Blend top and bottom images with a selectable mode and percentage."""

import warnings

import torch

if __package__ == "nodes":
    from utils.image_blend import BLEND_MODES, DEFAULT_DETAIL_SCALE, blend_image_batches
else:
    from ..utils.image_blend import BLEND_MODES, DEFAULT_DETAIL_SCALE, blend_image_batches

from .image_post_effects import _is_cuda_failure, resolve_post_effects_device


class MAIImageBlend:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = (
        "Blends top and bottom images with one mode and percentage across the batch. "
        "A single image repeats across the other batch. Two batches must have matching counts. "
        "For video upscale, connect the model to image_top and Lanczos to image_bottom. "
        "Uses CUDA when available and falls back to CPU."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_top": ("IMAGE", {"tooltip": "Top image or batch, blended over image_bottom."}),
                "image_bottom": ("IMAGE", {"tooltip": "Base image or batch. A 0% blend shows this input."}),
            },
            "optional": {
                "blend_mode": (list(BLEND_MODES), {"default": "normal"}),
                "blend_percentage": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01,
                    "tooltip": "Applies to the entire batch. 0% keeps the bottom image, and 100% applies the full blend mode.",
                }),
                "blend_list": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "hidden": True,
                    "tooltip": "Percentages saved by an older workflow. Clear this text to use blend_percentage.",
                }),
                "detail_scale": ("FLOAT", {
                    "default": DEFAULT_DETAIL_SCALE, "min": 0.1, "max": 32.0, "step": 0.1, "round": 0.1,
                    "tooltip": "Gaussian spread in output pixels for high_frequency and low_frequency. Larger values transfer broader detail in high_frequency and smoother tones in low_frequency.",
                }),
            },
        }

    def run(
        self, image_top, image_bottom, blend_list="", blend_mode="normal",
        blend_percentage=50.0, detail_scale=DEFAULT_DETAIL_SCALE,
    ):
        if not isinstance(image_bottom, torch.Tensor):
            raise ValueError("image_bottom must be a nonempty floating-point IMAGE batch [B, H, W, C].")

        compute_device = None
        try:
            compute_device = resolve_post_effects_device(image_bottom.device, "auto")
            output_device = image_bottom.device if compute_device.type == "cuda" else "cpu"
            return (blend_image_batches(
                image_top, image_bottom, blend_percentage, blend_mode,
                blend_list=blend_list, detail_scale=detail_scale,
                compute_device=compute_device, output_device=output_device,
            ),)
        except RuntimeError as exc:
            if (compute_device is not None and compute_device.type != "cuda") or not _is_cuda_failure(exc):
                raise
            warnings.warn(f"mAI image blend: CUDA failed; retrying on CPU. Reason: {exc}", RuntimeWarning, stacklevel=2)

        # Let the failed CUDA frame release its scratch tensors before retrying.
        return (blend_image_batches(
            image_top, image_bottom, blend_percentage, blend_mode,
            blend_list=blend_list, detail_scale=detail_scale,
            compute_device="cpu", output_device="cpu",
        ),)
