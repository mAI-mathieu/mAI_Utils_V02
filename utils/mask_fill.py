"""Mask fill operations without a ComfyUI runtime dependency."""

import math

import torch

from .mask_outline import parse_hex_color


def _prepare_inputs(image, mask):
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or min(image.shape) < 1
        or image.shape[-1] not in (3, 4)
        or not image.is_floating_point()
    ):
        raise ValueError("image must be a non-empty floating RGB or RGBA tensor [B,H,W,C]")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3 or min(mask.shape) < 1:
        raise ValueError("mask must have shape [H,W] or [B,H,W]")
    if tuple(mask.shape[1:]) != tuple(image.shape[1:3]):
        raise ValueError("mask dimensions must match the image dimensions")
    if image.shape[0] != mask.shape[0]:
        if image.shape[0] == 1:
            image = image.expand(mask.shape[0], -1, -1, -1)
        elif mask.shape[0] == 1:
            mask = mask.expand(image.shape[0], -1, -1)
        else:
            raise ValueError("image and mask batch sizes must match unless one batch is 1")
    if not torch.isfinite(image).all().item() or not torch.isfinite(mask).all().item():
        raise ValueError("image and mask must contain only finite values")
    # Keep soft mask precision even when the image uses half precision.
    work_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    return image, mask.to(device=image.device, dtype=work_dtype).clamp(0, 1)


def _load_opencv():
    # Loading the pack and using color fill never require OpenCV.
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Telea and Navier-Stokes fill require OpenCV. Install "
            "opencv-python-headless in the ComfyUI Python environment, then restart ComfyUI."
        ) from exc
    return cv2


def fill_masked_image(image, mask, method="color", color="#000000", inpaint_radius=3.0):
    """Fill white mask pixels, blend soft pixels, and preserve black pixels exactly.

    OpenCV operates on CPU with 8-bit RGB. Only its masked result is composited
    back; output device, floating dtype, and any alpha channel are preserved.
    """
    if method not in ("color", "telea", "navier_stokes"):
        raise ValueError(f"Unknown mask fill method: {method}")
    rgb_color = parse_hex_color(color) if method == "color" else None
    if method != "color" and (
        isinstance(inpaint_radius, bool)
        or not isinstance(inpaint_radius, (int, float))
        or not math.isfinite(inpaint_radius)
        or inpaint_radius < 1
    ):
        raise ValueError("inpaint_radius must be a finite number greater than or equal to 1")

    image, mask = _prepare_inputs(image, mask)
    output = image.clone()
    active = mask > 0
    if not active.any().item():
        return output
    if method != "color" and active.flatten(1).all(dim=1).any().item():
        raise ValueError(
            "Telea and Navier-Stokes need unmasked source pixels in each image; "
            "use color fill for a fully masked image."
        )

    if method == "color":
        fill = torch.tensor(rgb_color, dtype=mask.dtype, device=image.device)
        weights = mask.unsqueeze(-1)
        output[..., :3] = torch.where(
            active.unsqueeze(-1),
            torch.lerp(image[..., :3].to(mask.dtype), fill, weights).to(image.dtype),
            image[..., :3],
        )
        return output

    cv2 = _load_opencv()
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    for index in range(image.shape[0]):
        if not active[index].any().item():
            continue
        source = image[index, ..., :3].detach().to(device="cpu", dtype=torch.float32)
        source = (source.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy()
        binary_mask = (
            active[index].to(device="cpu", dtype=torch.uint8).contiguous().numpy()
        )
        # Both methods treat color channels independently, so RGB order is safe.
        filled = cv2.inpaint(source, binary_mask, float(inpaint_radius), flag)
        fill = torch.from_numpy(filled).to(device=image.device, dtype=mask.dtype) / 255.0
        blended = torch.lerp(
            image[index, ..., :3].to(mask.dtype), fill, mask[index, ..., None]
        )
        output[index, ..., :3] = torch.where(
            active[index, ..., None], blended.to(image.dtype), image[index, ..., :3]
        )
    return output
