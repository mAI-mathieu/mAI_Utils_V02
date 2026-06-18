from .nodes.example_text_node import MAIExampleTextNode
from .nodes.type_converter_node import MAITypeConverterNode

NODE_CLASS_MAPPINGS = {
    "MAIExampleTextNode": MAIExampleTextNode,
    "MAITypeConverterNode": MAITypeConverterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIExampleTextNode": "mAI Example Text Node",
    "MAITypeConverterNode": "mAI Type Converter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
