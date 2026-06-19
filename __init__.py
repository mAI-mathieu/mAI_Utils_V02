from .nodes.composite_layer_node import MAICompositeLayer
from .nodes.example_text_node import MAIExampleTextNode
from .nodes.type_converter_node import MAITypeConverterNode

NODE_CLASS_MAPPINGS = {
    "MAICompositeLayer": MAICompositeLayer,
    "MAIExampleTextNode": MAIExampleTextNode,
    "MAITypeConverterNode": MAITypeConverterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAICompositeLayer": "mAI Composite Layer",
    "MAIExampleTextNode": "mAI Example Text Node",
    "MAITypeConverterNode": "mAI Type Converter",
}

# MAIImageLayerStack remains in nodes/image_layer_stack_node.py, but is not
# registered by default because the compact chainable node replaces its tall UI.

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
