import torch


def create_bounding_box_mask(mask):
    """Return a filled bounding-box mask for each mask in the input batch."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")

    if mask.dim() == 2:
        masks = mask.unsqueeze(0)
        remove_batch_dimension = True
    elif mask.dim() == 3:
        masks = mask
        remove_batch_dimension = False
    else:
        raise ValueError(
            "mask must have shape [height, width] or [batch, height, width]"
        )

    output = torch.zeros_like(masks)

    for batch_index, current_mask in enumerate(masks):
        nonzero_pixels = torch.nonzero(current_mask > 0, as_tuple=False)
        if nonzero_pixels.numel() == 0:
            continue

        top, left = nonzero_pixels.min(dim=0).values.tolist()
        bottom, right = nonzero_pixels.max(dim=0).values.tolist()
        output[batch_index, top : bottom + 1, left : right + 1] = 1

    if remove_batch_dimension:
        return output[0]
    return output
