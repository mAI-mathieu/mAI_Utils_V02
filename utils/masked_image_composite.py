"""Mask-constrained image compositing without a ComfyUI runtime dependency."""

import math

import numpy as np
import torch
import torch.nn.functional as F


BLEND_MODES = ("feather", "content_aware_seam", "center_cut", "multiband")
MAX_SEAM_PIXELS = 262144


def _prepare_inputs(destination_image, source_image, source_mask):
    for name, image in (("destination_image", destination_image),
                        ("source_image", source_image)):
        if (not isinstance(image, torch.Tensor) or image.ndim != 4
                or min(image.shape) < 1 or image.shape[-1] not in (3, 4)
                or not image.is_floating_point()):
            raise ValueError(f"{name} must be a non-empty floating RGB/RGBA image [B,H,W,C]")
        if not torch.isfinite(image).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if destination_image.shape[1:] != source_image.shape[1:]:
        raise ValueError("source_image dimensions and channels must match destination_image")
    if not isinstance(source_mask, torch.Tensor):
        raise TypeError("source_mask must be a MASK tensor [H,W] or [B,H,W]")
    if source_mask.ndim == 2:
        source_mask = source_mask.unsqueeze(0)
    if (source_mask.ndim != 3 or min(source_mask.shape) < 1
            or source_mask.shape[1:] != destination_image.shape[1:3]):
        raise ValueError("source_mask spatial dimensions must match source_image")
    if not torch.isfinite(source_mask).all().item():
        raise ValueError("source_mask must contain only finite values")
    batch = max(destination_image.shape[0], source_image.shape[0], source_mask.shape[0])
    if any(t.shape[0] not in (1, batch) for t in (destination_image, source_image, source_mask)):
        raise ValueError("Image and source_mask batch sizes must match unless a batch is 1")
    dtype = torch.float64 if destination_image.dtype == torch.float64 else torch.float32
    destination = destination_image.expand(batch, -1, -1, -1)
    source = source_image.to(device=destination.device, dtype=destination.dtype).expand(batch, -1, -1, -1)
    mask = source_mask.to(device=destination.device, dtype=dtype).clamp(0, 1).expand(batch, -1, -1)
    return destination, source, mask


def _graph_cut(alpha, difference):
    """Four-neighbor min cut, with only feather pixels as graph vertices.

    Source-side vertices select source_image. Hard mask neighbors become
    terminal edges, so holes and disconnected regions need no special geometry.
    A weak unary mask prior favors the center when content costs are ambiguous.
    Capacities stay small enough for SciPy's integer flow implementation.
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order, maximum_flow
    except ImportError as exc:
        raise RuntimeError(
            "content_aware_seam requires SciPy (normally supplied by ComfyUI). "
            "Install scipy in the ComfyUI Python environment, then restart."
        ) from exc

    soft = (alpha > 0) & (alpha < 1)
    count = int(soft.sum())
    if count == 0:
        return alpha >= 0.5
    ids = np.full(alpha.shape, -1, dtype=np.int32)
    ids[soft] = np.arange(count, dtype=np.int32)
    source, sink = count, count + 1
    vertices = ids[soft]
    # Penalize choosing the opposite side of the supplied mask. A one-unit
    # source bias makes exactly 0.5 deterministic, matching center_cut.
    source_cost = np.rint(8 * alpha[soft]).astype(np.int32) + (alpha[soft] >= 0.5)
    sink_cost = np.rint(8 * (1 - alpha[soft])).astype(np.int32)
    rows = [np.full(count, source, dtype=np.int32), vertices]
    columns = [vertices, np.full(count, sink, dtype=np.int32)]
    capacities = [source_cost, sink_cost]

    def add_edges(first, second):
        first_ids, second_ids = ids[first], ids[second]
        # Low image disagreement makes an inexpensive seam. Clamp only the
        # analysis cost, never the output image or its protected endpoints.
        costs = np.rint(1024 * (0.001 + 0.5 * (
            difference[first] + difference[second]))).astype(np.int32)
        costs = np.maximum(costs, 1)
        both = (first_ids >= 0) & (second_ids >= 0)
        rows.extend((first_ids[both], second_ids[both]))
        columns.extend((second_ids[both], first_ids[both]))
        capacities.extend((costs[both], costs[both]))
        for own, other, other_alpha in ((first_ids, second_ids, alpha[second]),
                                       (second_ids, first_ids, alpha[first])):
            boundary = (own >= 0) & (other < 0)
            selected = own[boundary]
            from_source = other_alpha[boundary] >= 0.5
            weights = costs[boundary]
            rows.extend((np.full(int(from_source.sum()), source, dtype=np.int32),
                         selected[~from_source]))
            columns.extend((selected[from_source],
                            np.full(int((~from_source).sum()), sink, dtype=np.int32)))
            capacities.extend((weights[from_source], weights[~from_source]))

    add_edges((slice(None), slice(None, -1)), (slice(None), slice(1, None)))
    add_edges((slice(None, -1), slice(None)), (slice(1, None), slice(None)))
    graph = coo_matrix((np.concatenate(capacities),
                        (np.concatenate(rows), np.concatenate(columns))),
                       shape=(count + 2, count + 2), dtype=np.int32).tocsr()
    flow = maximum_flow(graph, source, sink)
    residual = graph - flow.flow
    residual.data = (residual.data > 0).astype(np.int32)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    source_side = np.zeros(count + 2, dtype=bool)
    source_side[reachable] = True
    result = alpha >= 0.5
    result[soft] = source_side[:count]
    return result


def _content_aware_alpha(destination, source, mask):
    """Bound the CPU graph size; lift labels back to the native feather band."""
    alpha = mask.detach().to(device="cpu", dtype=torch.float32)[None, None]
    difference = (destination[..., :3] - source[..., :3]).abs().mean(-1).clamp(0, 1)
    difference = difference.detach().to(device="cpu", dtype=torch.float32)[None, None]
    # Area reduction can create a few new feather pixels, so check again.
    while int(((alpha > 0) & (alpha < 1)).sum()) > MAX_SEAM_PIXELS:
        scale = min(0.9, math.sqrt(MAX_SEAM_PIXELS / float(((alpha > 0) & (alpha < 1)).sum())))
        height, width = alpha.shape[-2:]
        size = (max(1, int(height * scale)), max(1, int(width * scale)))
        alpha = F.interpolate(alpha, size=size, mode="area")
        difference = F.interpolate(difference, size=size, mode="area")
    labels = _graph_cut(alpha[0, 0].numpy(), difference[0, 0].numpy())
    selected = torch.from_numpy(labels.copy()).to(device=mask.device, dtype=mask.dtype)[None, None]
    if selected.shape[-2:] != mask.shape:
        selected = F.interpolate(selected, size=mask.shape, mode="nearest")
    # Never let a coarse search change the hard native mask.
    return torch.where(mask == 0, 0, torch.where(mask == 1, 1, selected[0, 0]))


def _pyramid_down(image):
    channels = image.shape[1]
    kernel = image.new_tensor([1, 4, 6, 4, 1]) / 16
    horizontal = kernel.view(1, 1, 1, 5).expand(channels, 1, 1, 5)
    vertical = kernel.view(1, 1, 5, 1).expand(channels, 1, 5, 1)
    blurred = F.conv2d(F.pad(image, (2, 2, 0, 0), mode="replicate"), horizontal, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (0, 0, 2, 2), mode="replicate"), vertical, groups=channels)
    return blurred[..., ::2, ::2]


def _pyramid_up(image, reference):
    return F.interpolate(image, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def _multiband(destination, source, mask, levels):
    """Laplacian color pyramids and a Gaussian mask pyramid, one image at a time."""
    destinations = [destination[..., :3].movedim(-1, 0)[None]]
    sources = [source[..., :3].movedim(-1, 0)[None]]
    masks = [mask[None, None]]
    for _ in range(levels - 1):
        if max(destinations[-1].shape[-2:]) == 1:
            break
        destinations.append(_pyramid_down(destinations[-1]))
        sources.append(_pyramid_down(sources[-1]))
        masks.append(_pyramid_down(masks[-1]))
    result = torch.lerp(destinations[-1], sources[-1], masks[-1])
    for index in range(len(destinations) - 2, -1, -1):
        dest_laplacian = destinations[index] - _pyramid_up(destinations[index + 1], destinations[index])
        src_laplacian = sources[index] - _pyramid_up(sources[index + 1], sources[index])
        result = _pyramid_up(result, destinations[index]) + torch.lerp(dest_laplacian, src_laplacian, masks[index])
    return result[0].movedim(0, -1)


@torch.no_grad()
def composite_with_source_mask(destination_image, source_image, source_mask,
                               blend_mode="feather", multiband_levels=5):
    """White takes source, black keeps destination, gray is the only seam area."""
    if blend_mode not in BLEND_MODES:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")
    if (isinstance(multiband_levels, bool) or not isinstance(multiband_levels, int)
            or not 1 <= multiband_levels <= 10):
        raise ValueError("multiband_levels must be an integer between 1 and 10")
    destination, source, masks = _prepare_inputs(destination_image, source_image, source_mask)
    output = destination.clone()
    # Half inputs compute in fp32, including under an enclosing autocast scope.
    with torch.autocast(device_type=destination.device.type, enabled=False):
        for index in range(destination.shape[0]):
            mask = masks[index]
            soft = (mask > 0) & (mask < 1)
            if not soft.any().item():
                output[index] = torch.where(mask[..., None] == 1, source[index], destination[index])
                continue
            dest = destination[index].to(mask.dtype)
            src = source[index].to(mask.dtype)
            if blend_mode == "center_cut":
                alpha = (mask >= 0.5).to(mask.dtype)
            elif blend_mode == "content_aware_seam":
                alpha = _content_aware_alpha(dest, src, mask)
            else:
                alpha = mask
            blended = torch.lerp(dest, src, alpha[..., None])
            if blend_mode == "multiband":
                color = _multiband(dest, src, mask, multiband_levels)
                # Fade the pyramid correction to zero at the hard mask edges.
                # Pyramid neighborhoods may read outside the feather; only gray
                # pixels receive corrections. RGBA alpha uses ordinary feather.
                envelope = (4 * mask * (1 - mask))[..., None]
                blended[..., :3] = torch.lerp(blended[..., :3], color, envelope).clamp(0, 1)
            blended = blended.to(output.dtype)
            if blend_mode in ("center_cut", "content_aware_seam"):
                # Hard cuts select native pixels exactly, avoiding lerp rounding.
                blended = torch.where(alpha[..., None] >= 0.5, source[index], destination[index])
            blended = torch.where(source[index] == destination[index], destination[index], blended)
            output[index] = torch.where(mask[..., None] == 0, destination[index],
                                        torch.where(mask[..., None] == 1, source[index], blended))
    return output
