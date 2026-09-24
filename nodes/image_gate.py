"""Conditionally supply an image to a downstream input that accepts None."""

_UNCONNECTED = object()


class MAIImageGate:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = (
        "Passes the image through when enabled; otherwise returns None. "
        "Connect only to inputs that accept None as a missing image. "
        "This does not remove the connection or block downstream execution."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True: pass the image. False: output None (missing image).",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "Image to pass through. Unconnected input produces None.",
                }),
            },
        }

    def check_lazy_status(self, enabled, image=_UNCONNECTED):
        # ComfyUI omits unconnected optional inputs, but supplies None for a
        # connected lazy input that has not been evaluated yet.
        return ["image"] if enabled and image is None else []

    def run(self, enabled, image=None):
        return (image if enabled else None,)
