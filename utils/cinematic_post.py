"""Display-referred cinematic finishing for ComfyUI RGB image tensors.

No ComfyUI imports. Spatial helpers use NCHW float32 tensors; the public entry
point accepts BHWC. Frames are processed separately to bound temporary memory.
"""

import math

import torch
import torch.nn.functional as F


# name: (default, minimum, maximum, step). Order also groups the node widgets.
FLOAT_CONTROLS = {
    # Master / tone
    "strength": (1.0, 0.0, 1.0, 0.01),
    "exposure": (0.0, -3.0, 3.0, 0.05),
    "contrast": (0.15, -0.5, 0.5, 0.01),
    "black_lift": (0.03, 0.0, 0.2, 0.005),
    "highlight_rolloff": (0.25, 0.0, 1.0, 0.01),
    "midtone_contrast": (0.15, -0.5, 0.5, 0.01),
    # Color
    "saturation": (-0.08, -1.0, 0.5, 0.01),
    "color_density": (0.12, 0.0, 0.5, 0.01),
    "shadow_cool": (0.08, 0.0, 0.5, 0.01),
    "highlight_warm": (0.08, 0.0, 0.5, 0.01),
    "green_tame": (0.06, 0.0, 0.5, 0.01),
    "blue_tame": (0.04, 0.0, 0.5, 0.01),
    "skin_protect": (0.20, 0.0, 1.0, 0.01),
    # Optical effects; blur sigma is in pixels at a 1080-pixel short edge.
    "halation_strength": (0.06, 0.0, 0.5, 0.01),
    "halation_threshold": (0.75, 0.0, 1.0, 0.01),
    "halation_blur": (6.0, 0.5, 40.0, 0.5),
    "bloom_strength": (0.04, 0.0, 0.5, 0.01),
    "bloom_threshold": (0.80, 0.0, 1.0, 0.01),
    "bloom_blur": (12.0, 0.5, 80.0, 0.5),
    "lens_softness": (0.08, 0.0, 0.5, 0.01),
    "sharpen_amount": (0.10, 0.0, 1.0, 0.01),
    "local_contrast": (0.08, 0.0, 0.5, 0.01),
    # Grain
    "grain_strength": (0.06, 0.0, 0.5, 0.01),
    "grain_size": (1.0, 0.5, 4.0, 0.1),
    "grain_chroma": (0.25, 0.0, 1.0, 0.01),
    # Lens finishing
    "vignette_strength": (0.08, 0.0, 0.5, 0.01),
    "vignette_feather": (0.75, 0.1, 1.0, 0.01),
    "chromatic_aberration": (0.25, 0.0, 2.0, 0.05),
    "lens_distortion": (0.0, -0.02, 0.02, 0.001),
}

# Presets replace the base look; widget differences from Subtle Film are trims.
PRESETS = {
    "Off / Neutral": {},
    "Subtle Film": {},
    "Commercial Cinematic": {
        "contrast": 0.22, "black_lift": 0.02, "color_density": 0.18,
        "shadow_cool": 0.12, "highlight_warm": 0.10, "grain_strength": 0.04,
    },
    "Commercial Cinematic - Preserve Colors": {
        "contrast": 0.22, "black_lift": 0.02, "color_density": 0.04,
        "saturation": 0.0, "shadow_cool": 0.0, "highlight_warm": 0.0,
        "green_tame": 0.0, "blue_tame": 0.0, "grain_strength": 0.04,
    },
    "Moody": {
        "exposure": -0.25, "contrast": 0.22, "saturation": -0.16,
        "shadow_cool": 0.16, "highlight_warm": 0.04, "vignette_strength": 0.16,
    },
    "Warm Premium": {
        "shadow_cool": 0.03, "highlight_warm": 0.16, "color_density": 0.16,
        "halation_strength": 0.08, "grain_strength": 0.04,
    },
    "Cool Night": {
        "exposure": -0.20, "shadow_cool": 0.20, "highlight_warm": 0.03,
        "saturation": -0.12, "blue_tame": 0.02, "bloom_strength": 0.07,
    },
}


def resolve_settings(preset="Subtle Film", **controls):
    if preset not in PRESETS:
        raise ValueError(f"Unknown cinematic preset: {preset}")
    unknown = controls.keys() - FLOAT_CONTROLS.keys()
    if unknown:
        raise ValueError(f"Unknown cinematic controls: {', '.join(sorted(unknown))}")
    settings = {}
    for name, (default, minimum, maximum, _step) in FLOAT_CONTROLS.items():
        value = float(controls.get(name, default))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        value = min(maximum, max(minimum, value))
        value += PRESETS[preset].get(name, default) - default
        settings[name] = min(maximum, max(minimum, value))
    return settings


def luminance(image):
    return (image * image.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)


def gaussian_blur(image, sigma):
    """Separable Gaussian, using reduced resolution for large radii.

    Replicate padding handles even 1x1 images. Kernels stay small regardless of
    image size. Area downsampling preserves isolated highlight energy.
    """
    if sigma <= 0:
        return image
    height, width = image.shape[-2:]
    reduction = max(1.0, sigma / 3.0)
    size = (max(1, math.ceil(height / reduction)), max(1, math.ceil(width / reduction)))
    work = image if size == (height, width) else F.interpolate(image, size=size, mode="area")
    channels = work.shape[1]
    for axis, scale in ((0, size[0] / height), (1, size[1] / width)):
        effective_sigma = max(0.1, sigma * scale)
        radius = min(12, max(1, math.ceil(3 * effective_sigma)))
        coordinate = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
        kernel = torch.exp(-0.5 * (coordinate / effective_sigma).square())
        kernel = kernel / kernel.sum()
        shape = (1, 1, -1, 1) if axis == 0 else (1, 1, 1, -1)
        padding = (0, 0, radius, radius) if axis == 0 else (radius, radius, 0, 0)
        weights = kernel.reshape(shape).expand(channels, 1, -1, -1)
        work = F.conv2d(F.pad(work, padding, mode="replicate"), weights, groups=channels)
    if size != (height, width):
        work = F.interpolate(work, size=(height, width), mode="bilinear", align_corners=False)
    return work


def compress_gamut(image):
    """Fit RGB around its luminance, retaining hue instead of clipping channels."""
    y = luminance(image).clamp(0, 1)
    chroma = image - y
    positive = chroma.amax(1, keepdim=True).clamp_min(1e-6)
    negative = (-chroma.amin(1, keepdim=True)).clamp_min(1e-6)
    scale = torch.minimum((1 - y) / positive, y / negative).clamp(0, 1)
    return y + chroma * scale


def apply_filmic_curve(image, exposure, contrast, black_lift, highlight_rolloff, midtone_contrast):
    exposed = image * (2.0 ** exposure)
    y = luminance(exposed)
    knee = 1.0 - 0.6 * highlight_rolloff
    headroom = max(1e-4, 1.0 - knee)
    shoulder = knee + headroom * (-torch.expm1(-(y - knee).clamp_min(0) / headroom))
    tone = torch.where(y > knee, shoulder, y)
    tone = tone + contrast * (tone - 0.5) * 4 * tone * (1 - tone)
    midtone_weight = 4 * tone * (1 - tone) * torch.exp(-((tone - 0.5) / 0.25).square())
    tone = tone + 0.5 * midtone_contrast * (tone - 0.5) * midtone_weight
    tone = tone + black_lift * (1 - tone).pow(3)
    # Scale chroma with luminance; adding the floor separately also lifts black.
    ratio = (tone / y.clamp_min(1e-6)).clamp(max=4.0)
    return compress_gamut(tone + (exposed - y) * ratio)


def subject_protection(image, mask):
    if mask is not None:
        return mask
    # A soft orange-hue heuristic, not face/skin detection. Neutral pixels are
    # excluded, and the modest skin_protect default limits false positives.
    red, green, blue = image.split(1, dim=1)
    chroma = image.amax(1, keepdim=True) - image.amin(1, keepdim=True)
    orange = torch.exp(-(((green - blue) / (red - blue).clamp_min(1e-5) - 0.45) / 0.30).square())
    red_dominance = ((red - green) / 0.08).clamp(0, 1)
    green_over_blue = ((green - blue) / 0.05).clamp(0, 1)
    return orange * red_dominance * green_over_blue * (chroma / 0.15).clamp(0, 1)


def apply_split_toning(image, shadow_cool, highlight_warm, protection):
    y = luminance(image)
    cool = image.new_tensor([-0.35, 0.05, 0.55])[None, :, None, None]
    warm = image.new_tensor([0.55, 0.12, -0.35])[None, :, None, None]
    tint = shadow_cool * (1 - y).square() * cool + highlight_warm * y.square() * warm
    tint = tint - luminance(tint)
    return image + tint * (4 * y * (1 - y)) * (1 - protection)


def apply_color_density(image, saturation, color_density, green_tame, blue_tame, protection):
    y = luminance(image)
    red, green, blue = image.split(1, dim=1)
    greens = ((green - torch.maximum(red, blue)) / 0.25).clamp(0, 1)
    blues = ((blue - torch.maximum(red, green)) / 0.25).clamp(0, 1)
    adjustment = saturation - green_tame * greens - blue_tame * blues
    colored = y + (image - y) * (1 + adjustment * (1 - protection))
    # Density darkens chromatic midtones like dye absorption, without boosting
    # channel distance from gray (ordinary saturation).
    chroma = image.amax(1, keepdim=True) - image.amin(1, keepdim=True)
    absorption = color_density * chroma * (4 * y * (1 - y)) * (1 - protection)
    return compress_gamut(colored * (1 - absorption))


def highlight_signal(image, threshold):
    return ((luminance(image) - threshold) / max(1e-6, 1 - threshold)).clamp(0, 1)


def apply_halation(image, strength, threshold, sigma):
    if strength == 0:
        return image
    highlights = highlight_signal(image, threshold)
    # Only the spread outside a bright source produces the red/orange halo.
    halo = (gaussian_blur(highlights, sigma) - highlights).clamp_min(0)
    tint = image.new_tensor([1.0, 0.28, 0.08])[None, :, None, None]
    return image + (1 - image) * halo * tint * strength


def apply_bloom(image, strength, threshold, sigma):
    if strength == 0:
        return image
    highlights = highlight_signal(image, threshold)
    glow = 0.65 * gaussian_blur(highlights, sigma) + 0.35 * gaussian_blur(highlights, sigma * 2)
    return image + (1 - image) * glow * strength


def apply_softness_and_sharpen(image, softness, sharpen, scale, subject):
    if softness:
        image = torch.lerp(image, gaussian_blur(image, max(0.4, 0.65 * scale)), softness)
    if sharpen:
        y = luminance(image)
        detail = y - gaussian_blur(y, max(0.5, 0.8 * scale))
        # Suppress low-amplitude noise and cap halos at high-contrast edges.
        gate = detail.abs() / (detail.abs() + 0.01)
        correction = detail.clamp(-0.06, 0.06) * gate * sharpen * (1 + 0.25 * subject)
        image = image + correction * (4 * y * (1 - y))
    return image


def apply_local_contrast(image, amount, scale, subject):
    if amount == 0:
        return image
    y = luminance(image)
    detail = (y - gaussian_blur(y, max(1.0, 12 * scale))).clamp(-0.15, 0.15)
    return image + detail * amount * (4 * y * (1 - y)) * (1 + 0.25 * subject)


def lens_coordinates(image):
    height, width = image.shape[-2:]
    x = (torch.arange(width, device=image.device, dtype=image.dtype) + 0.5) * (2 / width) - 1
    y = (torch.arange(height, device=image.device, dtype=image.dtype) + 0.5) * (2 / height) - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def apply_vignette(image, strength, feather):
    if strength == 0:
        return image
    xx, yy = lens_coordinates(image)
    radius = ((xx.square() + yy.square()) / 2).sqrt()
    edge = ((radius - (1 - feather)) / feather).clamp(0, 1)
    falloff = edge.square() * (3 - 2 * edge)
    return image * (1 - strength * falloff)


def apply_chromatic_aberration(image, amount, distortion, scale):
    if amount == 0 and distortion == 0:
        return image
    height, width = image.shape[-2:]
    xx, yy = lens_coordinates(image)
    radius_squared = (xx.square() + yy.square()) / 2
    base = 1 + distortion * radius_squared
    output = torch.empty_like(image)
    for channel, direction in enumerate((1, 0, -1)):
        # amount is approximately pixels per channel at a 1080p frame edge.
        dx = direction * amount * scale * 2 / width
        dy = direction * amount * scale * 2 / height
        grid = torch.stack((
            xx * (base + dx * radius_squared),
            yy * (base + dy * radius_squared),
        ), dim=-1)[None]
        output[:, channel:channel + 1] = F.grid_sample(
            image[:, channel:channel + 1], grid,
            mode="bilinear", padding_mode="border", align_corners=False,
        )
    return output


def apply_film_grain(image, strength, size, chroma, seed, scale):
    if strength == 0:
        return image
    height, width = image.shape[-2:]
    cell = max(1.0, size * scale)
    noise_size = (max(1, math.ceil(height / cell)), max(1, math.ceil(width / cell)))
    generator = torch.Generator(device=image.device).manual_seed(seed)
    noise = torch.randn((1, 4, *noise_size), generator=generator, device=image.device, dtype=image.dtype)
    if noise_size != (height, width):
        noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
    mono = noise[:, :1]
    color = noise[:, 1:] - noise[:, 1:].mean(1, keepdim=True)
    noise = (mono + chroma * color) / math.sqrt(1 + chroma * chroma * 2 / 3)
    y = luminance(image)
    envelope = (4 * y * (1 - y)).clamp_min(0).sqrt() * (1 - 0.35 * y)
    return image + noise * (strength * 0.10) * envelope


def normalize_mask(mask, batch, height, width, device):
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        raise TypeError("subject_mask must be a torch tensor")
    if mask.numel() == 0:
        return None
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.permute(0, 3, 1, 2)
    elif mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("subject_mask must be [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1]")
    if mask.shape[0] not in (1, batch):
        raise ValueError("subject_mask batch must be 1 or match the image batch")
    mask = mask.to(device=device, dtype=torch.float32)
    if not torch.isfinite(mask).all():
        raise ValueError("subject_mask must contain finite values")
    mask = mask.clamp(0, 1)
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return mask


@torch.no_grad()
def cinematic_post(image, subject_mask=None, enabled=True, preset="Subtle Film",
                   advanced_mode=False, grain_seed=0, grain_animation_safe=False, **controls):
    """Process BHWC float RGB on its existing device, preserving shape/dtype.

    advanced_mode controls frontend visibility only. Hidden controls retain
    their values; the toggle never disables processing or changes execution.
    """
    if not torch.is_tensor(image) or not torch.is_floating_point(image):
        raise TypeError("image must be a floating-point torch tensor")
    if image.ndim != 4 or image.shape[-1] != 3 or any(size == 0 for size in image.shape):
        raise ValueError("image must be a non-empty RGB tensor with shape [B,H,W,3]")
    settings = resolve_settings(preset, **controls)
    if not enabled or settings["strength"] == 0 or preset == "Off / Neutral":
        return image
    if not isinstance(grain_seed, int) or not 0 <= grain_seed <= 2**63 - 1:
        raise ValueError("grain_seed must be an integer between 0 and 2**63 - 1")
    batch, height, width, _channels = image.shape
    mask = normalize_mask(subject_mask, batch, height, width, image.device)
    scale = max(0.5, min(height, width) / 1080.0)
    result = torch.empty_like(image)
    p = settings
    for index in range(batch):
        original = image[index:index + 1].permute(0, 3, 1, 2).float()
        if not torch.isfinite(original).all():
            raise ValueError("image must contain finite values")
        original = original.clamp(0, 1)
        frame_mask = mask
        if mask is not None and mask.shape[0] > 1:
            frame_mask = mask[index:index + 1]
        subject = subject_protection(original, frame_mask)
        protection = subject * p["skin_protect"]
        work = apply_filmic_curve(
            original, p["exposure"], p["contrast"], p["black_lift"],
            p["highlight_rolloff"], p["midtone_contrast"],
        )
        work = apply_split_toning(work, p["shadow_cool"], p["highlight_warm"], protection)
        work = apply_color_density(
            work, p["saturation"], p["color_density"], p["green_tame"], p["blue_tame"], protection,
        )
        work = apply_halation(
            work, p["halation_strength"], p["halation_threshold"], p["halation_blur"] * scale,
        )
        work = apply_bloom(work, p["bloom_strength"], p["bloom_threshold"], p["bloom_blur"] * scale)
        work = apply_softness_and_sharpen(work, p["lens_softness"], p["sharpen_amount"], scale, protection)
        work = apply_local_contrast(work, p["local_contrast"], scale, protection)
        work = apply_vignette(work, p["vignette_strength"], p["vignette_feather"])
        work = apply_chromatic_aberration(work, p["chromatic_aberration"], p["lens_distortion"], scale)
        seed = (grain_seed + (index if grain_animation_safe else 0)) % (2**63)
        work = apply_film_grain(
            compress_gamut(work), p["grain_strength"], p["grain_size"], p["grain_chroma"], seed, scale,
        )
        work = torch.lerp(original, work.clamp(0, 1), p["strength"])
        result[index:index + 1] = work.permute(0, 2, 3, 1).to(dtype=image.dtype)
    return result
