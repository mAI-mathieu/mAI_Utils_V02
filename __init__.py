from .nodes.example_text_node import MAIExampleTextNode

NODE_CLASS_MAPPINGS = {
    "MAIExampleTextNode": MAIExampleTextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MAIExampleTextNode": "mAI Example Text Node",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
