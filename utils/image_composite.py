import torch
import torch.nn.functional as F


VALID_ANCHORS = {"top_left", "center"}
VALID_FIT_MODES = {"none", "contain", "cover"}


def _ensure_tensor(value, name):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch tensor")
    return value


def _ensure_float_tensor(value):
    if torch.is_floating_point(value):
        return value
    return value.float()


def _ensure_bhwc_image(image, name):
    image = _ensure_tensor(image, name)
    image = _ensure_float_tensor(image)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    if image.dim() != 4:
        raise ValueError(f"{name} must have shape [B, H, W, C]")

    if image.shape[-1] < 3:
        raise ValueError(f"{name} must have at least 3 channels")

    return image


def normalize_image_rgb(image, name="image"):
    image = _ensure_bhwc_image(image, name)
    return image[..., :3]


def _normalize_mask(mask, name="mask", height=None, width=None):
    mask = _ensure_tensor(mask, name)
    mask = _ensure_float_tensor(mask)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(-1)
    elif mask.dim() == 3:
        if height is not None and width is not None:
            if mask.shape[1] == height and mask.shape[2] == width:
                mask = mask.unsqueeze(-1)
            elif mask.shape[0] == height and mask.shape[1] == width:
                mask = mask[..., :1].unsqueeze(0)
            else:
                mask = mask.unsqueeze(-1)
        else:
            mask = mask.unsqueeze(-1)
    elif mask.dim() == 4:
        if mask.shape[-1] == 1:
            pass
        elif mask.shape[1] == 1:
            mask = mask.permute(0, 2, 3, 1)
        else:
            mask = mask[..., :1]
    else:
        raise ValueError(f"{name} must have shape [H, W], [B, H, W], or [B, H, W, 1]")

    return mask.clamp(0.0, 1.0)


def normalize_mask(mask, batch_size=None, height=None, width=None, name="mask", device=None, dtype=None):
    mask = _normalize_mask(mask, name, height=height, width=width)

    if device is not None or dtype is not None:
        mask = mask.to(device=device if device is not None else mask.device, dtype=dtype)

    if height is not None and width is not None:
        if mask.shape[1] != height or mask.shape[2] != width:
            mask = resize_tensor_mask(mask, width, height)

    if batch_size is not None:
        if mask.shape[0] == batch_size:
            return mask.clamp(0.0, 1.0)
        if mask.shape[0] == 1:
            return mask.repeat(batch_size, 1, 1, 1).clamp(0.0, 1.0)
        raise ValueError(f"{name} batch size is incompatible")

    return mask.clamp(0.0, 1.0)


def extract_layer_alpha(layer_image, layer_mask=None):
    layer_image = _ensure_bhwc_image(layer_image, "layer_image")

    if layer_mask is not None:
        return normalize_mask(
            layer_mask,
            height=layer_image.shape[1],
            width=layer_image.shape[2],
            name="layer_mask",
            device=layer_image.device,
            dtype=layer_image.dtype,
        )

    if layer_image.shape[-1] >= 4:
        return layer_image[..., 3:4].clamp(0.0, 1.0)

    return torch.ones(
        layer_image.shape[0],
        layer_image.shape[1],
        layer_image.shape[2],
        1,
        device=layer_image.device,
        dtype=layer_image.dtype,
    )


def extract_alpha(foreground, foreground_mask=None):
    return extract_layer_alpha(foreground, foreground_mask)


def resize_tensor_image(image, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than 0")

    image = _ensure_bhwc_image(image, "image")
    if image.shape[1] == height and image.shape[2] == width:
        return image

    channels_first = image.permute(0, 3, 1, 2)
    resized = F.interpolate(
        channels_first,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1)


def resize_tensor_mask(mask, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than 0")

    mask = _normalize_mask(mask, "mask")
    if mask.shape[1] == height and mask.shape[2] == width:
        return mask.clamp(0.0, 1.0)

    channels_first = mask.permute(0, 3, 1, 2)
    resized = F.interpolate(
        channels_first,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def resolve_batch(base_image, layer_image, alpha, base_mask=None):
    tensors = [
        ("base_image", base_image),
        ("layer_image", layer_image),
        ("alpha", alpha),
    ]
    if base_mask is not None:
        tensors.append(("base_mask", base_mask))

    batch_sizes = [tensor.shape[0] for _name, tensor in tensors]
    non_single_batches = {batch_size for batch_size in batch_sizes if batch_size != 1}

    if len(non_single_batches) > 1:
        raise ValueError(
            "background, foreground, and alpha batch sizes must match unless one is 1"
        )

    target_batch = next(iter(non_single_batches), 1)

    def repeat_to_target(tensor, name):
        if tensor.shape[0] == target_batch:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.repeat(target_batch, 1, 1, 1)
        raise ValueError(f"{name} batch size is incompatible")

    resolved = tuple(repeat_to_target(tensor, name) for name, tensor in tensors)
    return resolved


def compute_placement(bg_width, bg_height, fg_width, fg_height, x, y, anchor):
    if anchor not in VALID_ANCHORS:
        raise ValueError(f"Unknown anchor: {anchor}")

    if anchor == "top_left":
        left = int(x)
        top = int(y)
    else:
        left = int(round(float(x))) - fg_width // 2
        top = int(round(float(y))) - fg_height // 2

    right = left + fg_width
    bottom = top + fg_height

    overlap_left = max(0, left)
    overlap_top = max(0, top)
    overlap_right = min(bg_width, right)
    overlap_bottom = min(bg_height, bottom)

    if overlap_left >= overlap_right or overlap_top >= overlap_bottom:
        return None

    fg_left = overlap_left - left
    fg_top = overlap_top - top
    fg_right = fg_left + (overlap_right - overlap_left)
    fg_bottom = fg_top + (overlap_bottom - overlap_top)

    return {
        "background": (overlap_left, overlap_top, overlap_right, overlap_bottom),
        "foreground": (fg_left, fg_top, fg_right, fg_bottom),
    }


def _resolve_layer_size(bg_width, bg_height, fg_width, fg_height, scale, fit_mode):
    if fit_mode not in VALID_FIT_MODES:
        raise ValueError(f"Unknown fit_mode: {fit_mode}")

    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("scale must be greater than 0")

    fit_scale = 1.0
    if fit_mode == "contain":
        fit_scale = min(bg_width / fg_width, bg_height / fg_height)
    elif fit_mode == "cover":
        fit_scale = max(bg_width / fg_width, bg_height / fg_height)

    final_scale = fit_scale * scale
    width = max(1, int(round(fg_width * final_scale)))
    height = max(1, int(round(fg_height * final_scale)))
    return width, height


def composite_layer(
    base_image,
    layer_image,
    x,
    y,
    scale,
    opacity=1.0,
    anchor="top_left",
    fit_mode="none",
    base_mask=None,
    layer_mask=None,
):
    base_rgb = normalize_image_rgb(base_image, "base_image")
    layer_rgb = normalize_image_rgb(layer_image, "layer_image").to(
        device=base_rgb.device,
        dtype=base_rgb.dtype,
    )
    alpha = extract_layer_alpha(layer_image, layer_mask).to(
        device=base_rgb.device,
        dtype=base_rgb.dtype,
    )

    layer_height = layer_rgb.shape[1]
    layer_width = layer_rgb.shape[2]
    base_height = base_rgb.shape[1]
    base_width = base_rgb.shape[2]

    if alpha.shape[1] != layer_height or alpha.shape[2] != layer_width:
        alpha = resize_tensor_mask(alpha, layer_width, layer_height)

    resized_width, resized_height = _resolve_layer_size(
        base_width,
        base_height,
        layer_width,
        layer_height,
        scale,
        fit_mode,
    )

    layer_rgb = resize_tensor_image(layer_rgb, resized_width, resized_height)
    alpha = resize_tensor_mask(alpha, resized_width, resized_height)
    alpha = (alpha * float(opacity)).clamp(0.0, 1.0)

    if base_mask is not None:
        base_mask = normalize_mask(
            base_mask,
            height=base_height,
            width=base_width,
            name="base_mask",
            device=base_rgb.device,
            dtype=base_rgb.dtype,
        )
        base_rgb, layer_rgb, alpha, output_mask = resolve_batch(
            base_rgb,
            layer_rgb,
            alpha,
            base_mask,
        )
    else:
        base_rgb, layer_rgb, alpha = resolve_batch(base_rgb, layer_rgb, alpha)
        output_mask = torch.zeros(
            base_rgb.shape[0],
            base_rgb.shape[1],
            base_rgb.shape[2],
            1,
            device=base_rgb.device,
            dtype=base_rgb.dtype,
        )

    placement = compute_placement(
        base_rgb.shape[2],
        base_rgb.shape[1],
        layer_rgb.shape[2],
        layer_rgb.shape[1],
        x,
        y,
        anchor,
    )

    if placement is None:
        return base_rgb, output_mask[..., 0].clamp(0.0, 1.0)

    bg_left, bg_top, bg_right, bg_bottom = placement["background"]
    fg_left, fg_top, fg_right, fg_bottom = placement["foreground"]

    output = base_rgb.clone()
    mask_output = output_mask.clone()
    base_region = output[:, bg_top:bg_bottom, bg_left:bg_right, :]
    layer_region = layer_rgb[:, fg_top:fg_bottom, fg_left:fg_right, :]
    alpha_region = alpha[:, fg_top:fg_bottom, fg_left:fg_right, :]
    mask_region = mask_output[:, bg_top:bg_bottom, bg_left:bg_right, :]

    output[:, bg_top:bg_bottom, bg_left:bg_right, :] = (
        layer_region * alpha_region
        + base_region * (1.0 - alpha_region)
    )
    mask_output[:, bg_top:bg_bottom, bg_left:bg_right, :] = (
        alpha_region + mask_region * (1.0 - alpha_region)
    )
    return output.clamp(0.0, 1.0), mask_output[..., 0].clamp(0.0, 1.0)


def composite_images(
    background,
    foreground,
    x,
    y,
    scale,
    anchor,
    fit_mode,
    opacity=1.0,
    foreground_mask=None,
):
    image, _mask = composite_layer(
        background,
        foreground,
        x,
        y,
        scale,
        opacity=opacity,
        anchor=anchor,
        fit_mode=fit_mode,
        layer_mask=foreground_mask,
    )
    return image


def compose_layer_stack(background, layers):
    output = normalize_image_rgb(background, "background")

    for layer in layers:
        if not layer or layer.get("image") is None:
            continue

        def layer_value(name, default):
            value = layer.get(name, default)
            return default if value is None else value

        output = composite_images(
            output,
            layer["image"],
            layer_value("x", 0),
            layer_value("y", 0),
            layer_value("scale", 1.0),
            layer_value("anchor", "top_left"),
            layer_value("fit_mode", "none"),
            opacity=layer_value("opacity", 1.0),
            foreground_mask=layer.get("mask"),
        )

    return output
