"""Sharp edit compositing and conservative source-detail recovery for Auto."""

import torch

from .image_blur import blur_image
from .masked_image_composite import _content_aware_alpha


@torch.no_grad()
def recover_matching_detail(original, edited, mask):
    """Recover missing luminance detail only where native structure agrees.

    Inputs are validated, matching single-image BHWC tensors and a BHW mask.
    No geometry is warped. Broad edited color/lighting and RGBA alpha remain
    the base. Flat, displaced, unchanged and unselected areas receive no detail.
    """
    if not (mask == 1).any().item():
        return edited
    dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=original.device.type, enabled=False):
        first, second = original[..., :3].to(dtype), edited[..., :3].to(dtype)
        weights = first.new_tensor([0.2126, 0.7152, 0.0722])
        source_y = (first * weights).sum(-1, keepdim=True)
        edited_y = (second * weights).sum(-1, keepdim=True)
        source_fine, edited_fine = blur_image(source_y, 1), blur_image(edited_y, 1)
        source_mean, edited_mean = blur_image(source_fine, 2), blur_image(edited_fine, 2)
        source_var = (blur_image(source_fine.square(), 2) - source_mean.square()).clamp_min(0)
        edited_var = (blur_image(edited_fine.square(), 2) - edited_mean.square()).clamp_min(0)
        covariance = blur_image(source_fine * edited_fine, 2) - source_mean * edited_mean
        correlation = covariance / (source_var * edited_var).sqrt().clamp_min(1e-8)
        confidence = ((correlation - 0.9) / 0.1).clamp(0, 1)
        confidence *= (source_var > 1e-6) & (edited_var > 1e-6)
        # Adjust detail amplitude to the edit's luminance, so night edits do
        # not receive daytime-bright grain and recolors do not import old hue.
        gain = (blur_image(edited_y, 4) / blur_image(source_y, 4).clamp_min(0.02)).clamp(0, 2)
        target_detail = (source_y - source_fine) * gain
        edited_detail = edited_y - edited_fine
        target_energy = blur_image(target_detail.square(), 2).clamp_min(0).sqrt()
        edited_energy = blur_image(edited_detail.square(), 2).clamp_min(0).sqrt()
        detail_covariance = blur_image(target_detail * edited_detail, 2)
        detail_correlation = detail_covariance / (target_energy * edited_energy).clamp_min(1e-8)
        confidence *= ((detail_correlation - 0.7) / 0.3).clamp(0, 1)
        missing = ((1 - edited_energy / target_energy.clamp_min(1e-5) - 0.05) / 0.3).clamp(0, 1)
        correction = (target_detail - edited_detail) * confidence * missing * 0.75
        correction = correction.clamp(-0.02, 0.02)
        # A small shared RGB gain keeps the edited hue and bounds sharpening,
        # including saturated/dark colors where chromaticity can be extreme.
        relative_gain = (correction / edited_y.clamp_min(0.03)).clamp(-0.05, 0.05)
        corrected = (second * (1 + relative_gain)).clamp(0, 1)
        changed_pixel = (first != second).any(-1, keepdim=True)
        apply = (mask[..., None] == 1) & changed_pixel & (correction.abs() > 1e-8)
        result = edited.clone()
        result[..., :3] = torch.where(apply, corrected, second).to(edited.dtype)
    return result


@torch.no_grad()
def blend_edit_preserving_details(original, edited, masks, levels):
    """Feather subtle texture differences; hard-select strong displaced detail.

    This is a two-band decomposition: all detail above an automatically chosen
    coarse scale uses a content-aware selection where disagreement is strong.
    Low-contrast texture and the broad color layer follow the soft mask, avoiding
    a visible cut through walls or grain. Black/white endpoints stay native.
    Inputs are already validated and batch-matched by the node utility.
    """
    output = original.clone()
    dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
    sigma = float(max(4, min(16, 2 ** (levels - 1))))
    with torch.autocast(device_type=original.device.type, enabled=False):
        for index in range(original.shape[0]):
            mask = masks[index].to(dtype)
            if not ((mask > 0) & (mask < 1)).any().item():
                output[index] = torch.where(mask[..., None] == 1, edited[index], original[index])
                continue
            first, second = original[index].to(dtype), edited[index].to(dtype)
            alpha = _content_aware_alpha(first, second, mask)
            difference = second[..., :3] - first[..., :3]
            low_difference = blur_image(difference[None], sigma)[0]
            detail_difference = difference - low_difference
            # A hard cut through even a small residual creates a contour on a
            # flat background. Reserve hard selection for substantial detail
            # disagreement, where feathering could produce double object edges.
            # Smooth the evidence so individual grain pixels cannot switch the
            # blending rule. Max RGB also catches isoluminant colored edges.
            # Measure at a fine scale: coarse blur spill from a nearby changed
            # object must not turn a matching background into a sharp seam.
            fine_difference = difference - blur_image(difference[None], 1)[0]
            evidence = fine_difference.abs().amax(-1, keepdim=True)
            evidence = blur_image(evidence[None], 1)[0, ..., 0]
            sharpness = ((evidence - 0.006) / 0.012).clamp(0, 1)
            sharpness = sharpness.square() * (3 - 2 * sharpness)
            detail_alpha = torch.lerp(mask, alpha, sharpness)
            selected = torch.where(alpha[..., None] == 1, second, first)
            selected[..., :3] = (first[..., :3] + mask[..., None] * low_difference
                                 + detail_alpha[..., None] * detail_difference)
            selected[..., :3] = selected[..., :3].clamp(0, 1)
            selected = selected.to(output.dtype)
            selected = torch.where(original[index] == edited[index], original[index], selected)
            output[index] = torch.where(mask[..., None] == 0, original[index],
                                        torch.where(mask[..., None] == 1, edited[index], selected))
    return output
