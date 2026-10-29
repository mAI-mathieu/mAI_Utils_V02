"""Remove resize padding using its saved content rectangle."""

try:
    from ..utils.resize_stitch import recut_resize_padding
except ImportError:
    from utils.resize_stitch import recut_resize_padding


class MAIStitch:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("image", "width", "height", "mask")
    DESCRIPTION = (
        "Remove padding using the exact rectangle saved by mAI Fast GPU Resize. "
        "Connect its stitcher and the processed padded image; keep canvas dimensions unchanged. "
        "No resizing occurs. An optional padded-size mask is cropped identically; "
        "absent masks output zeros. Non-padding resize modes pass through the full image."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "stitcher": ("MAI_RESIZE_STITCHER",),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Padded-size mask to recut with the image. One mask broadcasts across the batch; omitted masks output zeros."}),
        }}

    def run(self, image, stitcher, mask=None):
        return recut_resize_padding(image, stitcher, mask)
