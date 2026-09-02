"""CropAndStitch-compatible crop node with an independent final blend mask."""

from __future__ import annotations

from copy import deepcopy

import torch

from ..utils.cropandstitch_dependency import load_cropandstitch_module
from ..utils.separate_stitch_mask import (
    broadcast_inputs,
    crop_mask_with_stitcher_geometry,
    extend_mask,
    full_crop_feather_mask,
    normalize_mask,
    preresize_dimensions,
    stack_stitcher_masks,
)


_UPSTREAM = load_cropandstitch_module()


class MAIInpaintCropSeparateStitchMask(_UPSTREAM.InpaintCropImproved):
    """Run upstream crop geometry while keeping render and stitch masks separate."""

    CATEGORY = "mAI / Image"
    DESCRIPTION = (
        "Crops for inpainting and generates a separate rectangular or expanded-mask "
        "blend region for final stitching."
    )
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = (
        "stitcher",
        "cropped_image",
        "cropped_mask",
        "cropped_stitch_mask",
    )

    @classmethod
    def INPUT_TYPES(cls):
        input_types = deepcopy(_UPSTREAM.InpaintCropImproved.INPUT_TYPES())
        input_types["required"]["stitch_mask_mode"] = (
            ["rectangle (full crop)", "extended mask"],
            {
                "default": "rectangle (full crop)",
                "tooltip": (
                    "Blend the whole crop, or use the render-mask shape expanded "
                    "by stitch_mask_expand_pixels."
                ),
            },
        )
        input_types["required"]["stitch_mask_expand_pixels"] = (
            "INT",
            {
                "default": 32,
                "min": 0,
                "max": _UPSTREAM.nodes.MAX_RESOLUTION,
                "step": 1,
                "tooltip": "Expansion used by extended mask mode.",
            },
        )
        input_types["required"]["stitch_mask_blend_pixels"] = (
            "INT",
            {
                "default": 32,
                "min": 0,
                "max": _UPSTREAM.nodes.MAX_RESOLUTION,
                "step": 1,
                "tooltip": "Feathering applied only to the separate stitch mask.",
            },
        )
        return input_types

    def inpaint_crop(
        self,
        stitch_mask_mode="rectangle (full crop)",
        stitch_mask_expand_pixels=32,
        stitch_mask_blend_pixels=32,
        **upstream_inputs,
    ):
        if stitch_mask_mode not in ("rectangle (full crop)", "extended mask"):
            raise ValueError(f"Unknown stitch mask mode: {stitch_mask_mode}")

        if stitch_mask_mode == "rectangle (full crop)":
            upstream_result = super().inpaint_crop(**upstream_inputs)
            stitcher, cropped_image, cropped_mask = upstream_result[:3]
            processor = self._processor_for(stitcher["device_mode"])
            processing_device = self._processing_device(stitcher["device_mode"])
            processed_stitch_masks = []
            for _ in range(cropped_image.shape[0]):
                # Rectangle mode deliberately covers the complete crop canvas.
                full_crop_mask = full_crop_feather_mask(
                    1,
                    cropped_image.shape[1],
                    cropped_image.shape[2],
                    int(stitch_mask_blend_pixels),
                    processing_device,
                    torch.float32,
                )
                processed_stitch_masks.append(full_crop_mask.cpu())

            stitcher["cropped_mask_for_blend"] = processed_stitch_masks
            return (
                stitcher,
                cropped_image,
                cropped_mask,
                stack_stitcher_masks(stitcher),
            )

        stitch_source_mask = upstream_inputs.get("mask")
        if stitch_source_mask is None:
            stitch_source_mask = torch.zeros_like(upstream_inputs["image"][:, :, :, 0])

        image, render_mask, context_mask, stitch_source_mask = broadcast_inputs(
            upstream_inputs["image"],
            upstream_inputs.get("mask"),
            upstream_inputs.get("optional_context_mask"),
            stitch_source_mask,
        )

        processor = self._processor_for(upstream_inputs["device_mode"])
        processing_device = self._processing_device(upstream_inputs["device_mode"])
        stitch_mask = stitch_source_mask.to(processing_device)
        if stitch_mask_expand_pixels > 0:
            stitch_mask = processor.expand_m(stitch_mask, int(stitch_mask_expand_pixels))
        stitch_mask = stitch_mask.float().clamp(0.0, 1.0).cpu()

        # render_mask: sent to the model. crop_area_mask: used only for geometry.
        # stitch_mask: render-mask shape expanded independently for compositing.
        stitch_mask_for_bbox = (stitch_mask > 0).to(
            device=context_mask.device, dtype=context_mask.dtype
        )
        crop_area_mask = torch.maximum(context_mask, stitch_mask_for_bbox)

        delegated_inputs = dict(upstream_inputs)
        delegated_inputs["image"] = image
        delegated_inputs["mask"] = render_mask
        delegated_inputs["optional_context_mask"] = crop_area_mask
        upstream_result = super().inpaint_crop(**delegated_inputs)
        stitcher, cropped_image, cropped_mask = upstream_result[:3]

        processed_stitch_masks = []
        for index in range(stitch_mask.shape[0]):
            one_mask = normalize_mask(stitch_mask[index], "stitch_mask").to(
                self._processing_device(stitcher["device_mode"])
            )
            one_mask = self._preprocess_stitch_mask(one_mask, processor, upstream_inputs)
            geometry = self._geometry_for(stitcher, index)
            transformed = crop_mask_with_stitcher_geometry(
                one_mask,
                geometry,
                cropped_image.shape[1],
                cropped_image.shape[2],
                processor,
                stitcher["downscale_algorithm"],
                stitcher["upscale_algorithm"],
                int(stitch_mask_blend_pixels),
            ).cpu()
            processed_stitch_masks.append(transformed)

        stitcher["cropped_mask_for_blend"] = processed_stitch_masks
        cropped_stitch_mask = stack_stitcher_masks(stitcher)
        return stitcher, cropped_image, cropped_mask, cropped_stitch_mask

    @staticmethod
    def _processor_for(device_mode):
        if device_mode == "gpu (much faster)":
            return _UPSTREAM.GPUProcessorLogic()
        return _UPSTREAM.CPUProcessorLogic()

    @staticmethod
    def _processing_device(device_mode):
        if device_mode == "gpu (much faster)":
            return _UPSTREAM.comfy.model_management.get_torch_device()
        return torch.device("cpu")

    @staticmethod
    def _preprocess_stitch_mask(mask, processor, inputs):
        if inputs["preresize"]:
            target_width, target_height, algorithm = preresize_dimensions(
                mask.shape[2],
                mask.shape[1],
                inputs["preresize_mode"],
                inputs["preresize_min_width"],
                inputs["preresize_min_height"],
                inputs["preresize_max_width"],
                inputs["preresize_max_height"],
            )
            if algorithm is not None:
                mask = processor.rescale_m(mask, target_width, target_height, algorithm)

        if inputs["extend_for_outpainting"]:
            mask = extend_mask(
                mask,
                inputs["extend_up_factor"],
                inputs["extend_down_factor"],
                inputs["extend_left_factor"],
                inputs["extend_right_factor"],
            )
        return mask

    @staticmethod
    def _geometry_for(stitcher, index):
        canvas = stitcher["canvas_image"][index]
        return {
            "canvas_shape": (canvas.shape[1], canvas.shape[2]),
            "canvas_to_orig_x": stitcher["canvas_to_orig_x"][index],
            "canvas_to_orig_y": stitcher["canvas_to_orig_y"][index],
            "canvas_to_orig_w": stitcher["canvas_to_orig_w"][index],
            "canvas_to_orig_h": stitcher["canvas_to_orig_h"][index],
            "cropped_to_canvas_x": stitcher["cropped_to_canvas_x"][index],
            "cropped_to_canvas_y": stitcher["cropped_to_canvas_y"][index],
            "cropped_to_canvas_w": stitcher["cropped_to_canvas_w"][index],
            "cropped_to_canvas_h": stitcher["cropped_to_canvas_h"][index],
        }
