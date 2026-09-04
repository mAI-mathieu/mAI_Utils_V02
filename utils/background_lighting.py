import torch


def _normalize_image(image, name):
    if not torch.is_tensor(image):
        raise TypeError(f"{name} must be a torch tensor")

    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[-1] < 3:
        raise ValueError(f"{name} must have shape [B, H, W, C] with at least 3 channels")

    if not torch.is_floating_point(image):
        image = image.float()
    return image[..., :3]


def _normalize_mask(mask, height, width, device, dtype):
    if mask is None:
        return torch.zeros((1, height, width, 1), device=device, dtype=dtype)

    if not torch.is_tensor(mask):
        raise TypeError("edit_mask must be a torch tensor")

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(-1)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(-1)
    elif mask.dim() == 4 and mask.shape[-1] == 1:
        pass
    elif mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.permute(0, 2, 3, 1)
    else:
        raise ValueError(
            "edit_mask must have shape [H, W], [B, H, W], or [B, H, W, 1]"
        )

    if mask.shape[1] != height or mask.shape[2] != width:
        raise ValueError("edit_mask dimensions must match the image dimensions")

    return mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _resolve_batch(*named_tensors):
    non_single_batches = {
        tensor.shape[0]
        for _name, tensor in named_tensors
        if tensor.shape[0] != 1
    }
    if len(non_single_batches) > 1:
        names = ", ".join(name for name, _tensor in named_tensors)
        raise ValueError(f"{names} batch sizes must match unless one is 1")

    target_batch = next(iter(non_single_batches), 1)
    resolved = []
    for name, tensor in named_tensors:
        if tensor.shape[0] == target_batch:
            resolved.append(tensor)
        elif tensor.shape[0] == 1:
            resolved.append(tensor.expand(target_batch, -1, -1, -1))
        else:
            raise ValueError(f"{name} batch size is incompatible")
    return tuple(resolved)


def _weighted_mean_and_std(image, weights, epsilon):
    weight_sum = weights.sum(dim=(1, 2), keepdim=True)
    if torch.any(weight_sum <= epsilon):
        raise ValueError(
            "edit_mask leaves no untouched background pixels for lighting analysis"
        )

    mean = (image * weights).sum(dim=(1, 2), keepdim=True) / weight_sum
    variance = (
        ((image - mean) ** 2 * weights).sum(dim=(1, 2), keepdim=True)
        / weight_sum
    )
    return mean, variance.clamp_min(0.0).sqrt()


def match_background_lighting(original_image, edited_image, edit_mask=None, epsilon=1e-6):
    """Match edited RGB channel mean/std using only inverse-mask pixels.

    ComfyUI masks use 1 for the edited region. The complement therefore weights
    the untouched background. The derived affine correction is applied to every
    pixel of the edited image and the result is constrained to ComfyUI's image
    range. When no mask is supplied, the whole image is used for analysis.
    """
    original = _normalize_image(original_image, "original_image")
    edited = _normalize_image(edited_image, "edited_image")

    if original.shape[1:3] != edited.shape[1:3]:
        raise ValueError("original_image and edited_image dimensions must match")

    compute_dtype = (
        torch.float32
        if edited.dtype in (torch.float16, torch.bfloat16)
        else edited.dtype
    )
    edited = edited.to(dtype=compute_dtype)
    original = original.to(device=edited.device, dtype=compute_dtype)
    mask = _normalize_mask(
        edit_mask,
        edited.shape[1],
        edited.shape[2],
        edited.device,
        compute_dtype,
    )
    original, edited, mask = _resolve_batch(
        ("original_image", original),
        ("edited_image", edited),
        ("edit_mask", mask),
    )

    background_weights = 1.0 - mask
    original_mean, original_std = _weighted_mean_and_std(
        original, background_weights, epsilon
    )
    edited_mean, edited_std = _weighted_mean_and_std(
        edited, background_weights, epsilon
    )

    impossible_channels = (edited_std <= epsilon) & (original_std > epsilon)
    if torch.any(impossible_channels):
        raise ValueError(
            "edited untouched background has zero contrast, so the original "
            "contrast cannot be recovered with an affine lighting correction"
        )

    contrast_scale = torch.where(
        edited_std > epsilon,
        original_std / edited_std.clamp_min(epsilon),
        torch.ones_like(edited_std),
    )
    corrected = (edited - edited_mean) * contrast_scale + original_mean
    return corrected.clamp(0.0, 1.0)
