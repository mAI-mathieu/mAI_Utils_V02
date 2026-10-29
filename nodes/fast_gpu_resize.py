"""ComfyUI wrapper for batched, tensor-only resizing."""

import torch

try:
    from ..utils.resize_kernels import RESIZE_METHODS, RESIZE_MODES, resize_image_batch
    from ..utils.resize_masks import prepare_resize_mask, resize_mask_batch
    from ..utils.resize_stitch import create_resize_stitcher
except ImportError:
    from utils.resize_kernels import RESIZE_METHODS, RESIZE_MODES, resize_image_batch
    from utils.resize_masks import prepare_resize_mask, resize_mask_batch
    from utils.resize_stitch import create_resize_stitcher


def resolve_compute_device(image_device, device):
    if device == "auto":
        return image_device
    if device == "cpu":
        return torch.device("cpu")
    if device != "gpu":
        raise ValueError(f"Unknown device: {device}")
    if not torch.cuda.is_available():
        raise ValueError("GPU resizing requires an available CUDA device; choose auto or cpu.")
    # Import only during explicit GPU execution, never while loading the pack.
    try:
        from comfy.model_management import get_torch_device
    except ModuleNotFoundError as exc:
        if exc.name not in ("comfy", "comfy.model_management"):
            raise
        preferred = torch.device("cuda", torch.cuda.current_device())
    else:
        preferred = get_torch_device()
    if preferred.type != "cuda":
        raise ValueError("ComfyUI's preferred device is not CUDA; choose auto or cpu.")
    return preferred


class MAIFastGPUResize:
    CATEGORY = "mAI / Image"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "MASK", "MAI_RESIZE_STITCHER")
    RETURN_NAMES = ("image", "width", "height", "method_used", "mask", "stitcher")
    DESCRIPTION = (
        "Batched torch resize with native, Lanczos and BC cubic kernels. "
        "Auto device keeps the input device; select gpu to move CPU images to CUDA. "
        "Keep aspect fits within width/height without padding or cropping. "
        "Fit pads black; fill crops from the center. "
        "Pad to multiple keeps the original image size and only adds black padding. "
        "Center crop takes the exact width/height from the original without resizing. "
        "Pad to multiple edge extends the border pixels instead of padding black. "
        "These padding/crop modes preserve source pixels and dtype exactly; "
        "method, antialias and precision are ignored. "
        "An optional source-size mask follows the same transform; absent masks output zeros. "
        "Connect stitcher to mAI stitch to remove padding later at the exact saved position."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 1280, "min": 1, "max": 16384}),
            "height": ("INT", {"default": 720, "min": 1, "max": 16384}),
            "resize_mode": (list(RESIZE_MODES), {"default": "exact",
                                               "tooltip": "keep_aspect fits within width/height without padding or cropping. pad_to_multiple and pad_to_multiple_edge ignore width/height and preserve original image size, padding to the next multiple_of with black or stretched edge pixels. center_crop crops the original to exact width/height without resizing; dimensions must not exceed the source."}),
            "method": (list(RESIZE_METHODS), {"default": "auto",
                                             "tooltip": "Interpolation method. Ignored in pad_to_multiple, pad_to_multiple_edge and center_crop; original pixels are copied exactly."}),
            "antialias": ("BOOLEAN", {"default": True}),
            "multiple_of": ("INT", {"default": 1, "min": 1, "max": 16384, "step": 1,
                                    "tooltip": "Round output width and height to the nearest multiple; ties round upward. pad_to_multiple and pad_to_multiple_edge only pad original dimensions upward, leaving aligned dimensions unchanged. 1 disables rounding/padding. Ignored in center_crop mode so width/height stay exact."}),
            "device": (["auto", "gpu", "cpu"], {"default": "auto"}),
            "precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto",
                                                                    "tooltip": "Output dtype for resizing. Ignored in pad_to_multiple, pad_to_multiple_edge and center_crop to preserve original pixel values and dtype."}),
            "chunk_size": ("INT", {"default": 0, "min": 0, "max": 1000000,
                                   "tooltip": "0: whole batch for native modes, automatic memory-bounded chunks for custom kernels. Positive: frames per chunk."}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Source-size mask transformed with the image. One mask broadcasts across an image batch. If omitted, outputs a zero mask at the output size."}),
        }}

    def run(self, image, width=1280, height=720, resize_mode="exact", method="auto",
            antialias=True, multiple_of=1, device="auto", precision="auto", chunk_size=0, mask=None):
        if not isinstance(image, torch.Tensor):
            raise ValueError("image must be a ComfyUI IMAGE tensor.")
        mask = prepare_resize_mask(mask, image.shape)
        result = resize_image_batch(
            image, width, height, resize_mode, method, antialias, multiple_of,
            resolve_compute_device(image.device, device), precision, chunk_size,
        )
        output_mask = resize_mask_batch(
            mask, result[0], width, height, resize_mode, method, antialias,
            multiple_of, precision, chunk_size,
        )
        stitcher = create_resize_stitcher(
            image.shape[2], image.shape[1], width, height, resize_mode, multiple_of,
        )
        return (*result, output_mask, stitcher)
