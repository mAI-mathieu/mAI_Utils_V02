from .nodes.background_lighting_match import MAIBackgroundLightingMatch
from .nodes.cinematic_post import mAI_CinematicPost
from .nodes.composite_layer_node import MAICompositeLayer
from .nodes.conditional_lora import MAIConditionalLora
from .nodes.example_text_node import MAIExampleTextNode
from .nodes.fast_gpu_resize import MAIFastGPUResize
from .nodes.frame_loop_fade import MAIFrameLoopFade
from .nodes.gpu_video_combine import MAIGPUVideoCombine
from .nodes.h3_to_ltx_frame_adapter import MAIH3ToLTXFrameAdapter
from .nodes.image_aspect_ratio import MAIImageAspectRatio
from .nodes.image_gate import MAIImageGate
from .nodes.image_logic_check import MAIImageLogicCheck
from .nodes.json_parser import MAIJsonParser
from .nodes.krea2_image_conditioning import MAIKrea2ImageConditioning
from .nodes.mask_bounding_box import MAIMaskBoundingBox
from .nodes.mask_outline import MAIMaskOutline
from .nodes.prepare_image_for_minimax_h3 import MAIPrepareImageForMinimaxH3
from .nodes.random_line_node import MAIRandomLine
from .nodes.save_text_file_node import MAISaveTextFile
from .nodes.text_sequence_randomizer import MAITextSequenceRandomizer
from .nodes.trim_frame_sequence import MAITrimFrameSequence
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
    "mAI_CinematicPost": mAI_CinematicPost,
    "MAIBackgroundLightingMatch": MAIBackgroundLightingMatch,
    "MAICompositeLayer": MAICompositeLayer,
    "MAIConditionalLora": MAIConditionalLora,
    "MAIExampleTextNode": MAIExampleTextNode,
    "MAIFastGPUResize": MAIFastGPUResize,
    "MAIFrameLoopFade": MAIFrameLoopFade,
    "MAIGPUVideoCombine": MAIGPUVideoCombine,
    "MAIH3ToLTXFrameAdapter": MAIH3ToLTXFrameAdapter,
    "MAIImageAspectRatio": MAIImageAspectRatio,
    "MAIImageGate": MAIImageGate,
    "MAIImageLogicCheck": MAIImageLogicCheck,
    "MAIJsonParser": MAIJsonParser,
    "MAIKrea2ImageConditioning": MAIKrea2ImageConditioning,
    "MAIMaskBoundingBox": MAIMaskBoundingBox,
    "MAIMaskOutline": MAIMaskOutline,
    "MAIPrepareImageForMinimaxH3": MAIPrepareImageForMinimaxH3,
    "MAIRandomLine": MAIRandomLine,
    "MAISaveTextFile": MAISaveTextFile,
    "MAITextSequenceRandomizer": MAITextSequenceRandomizer,
    "MAITrimFrameSequence": MAITrimFrameSequence,
    "MAITypeConverterNode": MAITypeConverterNode,
    "MAIVideoLoader": MAIVideoLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mAI_CinematicPost": "mAI Cinematic Post",
    "MAIBackgroundLightingMatch": "mAI background lighting match",
    "MAICompositeLayer": "mAI Composite Layer",
    "MAIConditionalLora": "mAI conditional LoRA",
    "MAIExampleTextNode": "mAI Example Text Node",
    "MAIFastGPUResize": "mAI Fast GPU Resize",
    "MAIFrameLoopFade": "mAI frame loop fade",
    "MAIGPUVideoCombine": "mAI GPU Video Combine",
    "MAIH3ToLTXFrameAdapter": "mAI H3 to LTX Frame Adapter",
    "MAIImageAspectRatio": "mAI image aspect ratio",
    "MAIImageGate": "mAI image gate",
    "MAIImageLogicCheck": "mAI Image Logic Check",
    "MAIJsonParser": "mAI JSON parser",
    "MAIKrea2ImageConditioning": "mAI Krea2 image conditioning",
    "MAIMaskBoundingBox": "mAI mask bounding box",
    "MAIMaskOutline": "mAI mask outline",
    "MAIPrepareImageForMinimaxH3": "mAI prepare image for Minimax H3",
    "MAIRandomLine": "mAI Random Line",
    "MAISaveTextFile": "mAI Save Text File",
    "MAITextSequenceRandomizer": "mAI text sequence randomizer",
    "MAITrimFrameSequence": "mAI trim frame sequence",
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

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
