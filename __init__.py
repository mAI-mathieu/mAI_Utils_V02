from .nodes.background_lighting_match import MAIBackgroundLightingMatch
from .nodes.composite_layer_node import MAICompositeLayer
from .nodes.example_text_node import MAIExampleTextNode
from .nodes.mask_bounding_box import MAIMaskBoundingBox
from .nodes.mask_outline import MAIMaskOutline
from .nodes.prepare_image_for_minimax_h3 import MAIPrepareImageForMinimaxH3
from .nodes.random_line_node import MAIRandomLine
from .nodes.save_text_file_node import MAISaveTextFile
from .nodes.text_sequence_randomizer import MAITextSequenceRandomizer
from .nodes.type_converter_node import MAITypeConverterNode
from .nodes.video_loader import MAIVideoLoader
from .utils.cropandstitch_dependency import CropAndStitchDependencyError

try:
    from .nodes.inpaint_crop_separate_stitch_mask import (
        MAIInpaintCropSeparateStitchMask,
    )
except CropAndStitchDependencyError as exc:
    MAIInpaintCropSeparateStitchMask = None
    if exc.missing:
        print(
            "[mAI] mAI Inpaint Crop - Separate Stitch Mask was not loaded "
            "because ComfyUI-Inpaint-CropAndStitch is not installed."
        )
    else:
        print(
            "[mAI] mAI Inpaint Crop - Separate Stitch Mask was not loaded: "
            f"{exc}"
        )

NODE_CLASS_MAPPINGS = {
    "MAIBackgroundLightingMatch": MAIBackgroundLightingMatch,
    "MAICompositeLayer": MAICompositeLayer,
    "MAIExampleTextNode": MAIExampleTextNode,
    "MAIMaskBoundingBox": MAIMaskBoundingBox,
    "MAIMaskOutline": MAIMaskOutline,
    "MAIPrepareImageForMinimaxH3": MAIPrepareImageForMinimaxH3,
    "MAIRandomLine": MAIRandomLine,
    "MAISaveTextFile": MAISaveTextFile,
    "MAITextSequenceRandomizer": MAITextSequenceRandomizer,
    "MAITypeConverterNode": MAITypeConverterNode,
    "MAIVideoLoader": MAIVideoLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIBackgroundLightingMatch": "mAI background lighting match",
    "MAICompositeLayer": "mAI Composite Layer",
    "MAIExampleTextNode": "mAI Example Text Node",
    "MAIMaskBoundingBox": "mAI mask bounding box",
    "MAIMaskOutline": "mAI mask outline",
    "MAIPrepareImageForMinimaxH3": "mAI prepare image for Minimax H3",
    "MAIRandomLine": "mAI Random Line",
    "MAISaveTextFile": "mAI Save Text File",
    "MAITextSequenceRandomizer": "mAI text sequence randomizer",
    "MAITypeConverterNode": "mAI Type Converter",
    "MAIVideoLoader": "mAI video loader",
}

if MAIInpaintCropSeparateStitchMask is not None:
    NODE_CLASS_MAPPINGS["MAIInpaintCropSeparateStitchMask"] = (
        MAIInpaintCropSeparateStitchMask
    )
    NODE_DISPLAY_NAME_MAPPINGS["MAIInpaintCropSeparateStitchMask"] = (
        "mAI Inpaint Crop - Separate Stitch Mask"
    )

# MAIImageLayerStack remains in nodes/image_layer_stack_node.py, but is not
# registered by default because the compact chainable node replaces its tall UI.

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
