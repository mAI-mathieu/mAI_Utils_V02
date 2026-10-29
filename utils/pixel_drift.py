"""Conservative feature-based image registration, independent of ComfyUI.

Only the edited image is resampled. The reference supplies geometry, never pixels.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class AlignmentDecision:
    matrix: np.ndarray | None = None
    reason: str = "insufficient matching features"
    similarity: float = 0.0
    inliers: int = 0
    model: str = "perspective"


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "mAI pixel drift fix requires OpenCV with SIFT. "
            "Install this pack's requirements in the ComfyUI Python environment."
        ) from exc
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("mAI pixel drift fix requires OpenCV 4.5 or newer with SIFT.")
    return cv2


def _spread(points, width, height, cv2):
    """A small logo or clustered object cannot justify a whole-image warp."""
    normalized = points / np.array([width, height], dtype=np.float32)
    area = cv2.contourArea(cv2.convexHull(normalized.astype(np.float32)))
    cells = np.clip((normalized * 3).astype(int), 0, 2)
    return area >= 0.20 and len(np.unique(cells, axis=0)) >= 4


def _safe_transform(matrix, width, height, max_drift, cv2):
    """Reject poles, flips, collapsed quads, and excessive corner movement."""
    if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-8:
        return False
    matrix = matrix / matrix[2, 2]
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    denominators = corners @ matrix[2, :2] + 1
    if np.min(denominators) <= 0.25 or np.max(denominators) >= 4:
        return False
    projected = cv2.perspectiveTransform(corners[None], matrix)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return False
    area_ratio = cv2.contourArea(projected, oriented=True) / ((width - 1) * (height - 1))
    movement = np.abs(projected - corners) / np.array([width - 1, height - 1])
    return 0.65 <= area_ratio <= 1.50 and float(movement.max()) <= max_drift


def _structure_scores(reference, candidate, valid, cv2):
    """Local normalized correlations tolerate color/contrast edits and local replacements.

    The denominator includes all usable tiles, including flat/nonmatching ones,
    so a shared patch cannot masquerade as whole-image similarity.
    """
    reference = cv2.GaussianBlur(reference.astype(np.float32), (0, 0), 1.0)
    candidate = cv2.GaussianBlur(candidate.astype(np.float32), (0, 0), 1.0)
    height, width = reference.shape
    scores = []
    for rows in np.array_split(np.arange(height), 8):
        for cols in np.array_split(np.arange(width), 8):
            region = np.ix_(rows, cols)
            keep = valid[region]
            if keep.mean() < 0.80:
                continue
            a, b = reference[region][keep], candidate[region][keep]
            a, b = a - a.mean(), b - b.mean()
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if a.std() < 2.0 or b.std() < 2.0 or denominator < 1e-6:
                scores.append(0.0)
            else:
                scores.append(float(np.clip(np.dot(a, b) / denominator, -1, 1)))
    return np.asarray(scores)


def _translation_fit(edit_points, ref_points):
    """Prefer a robust translation when feature motion is uniform.

    A median rejects changed objects. The tolerance follows the measured feature
    noise, with a tight cap to avoid explaining rotation or scale as a shift.
    """
    offsets = ref_points - edit_points
    shift = np.median(offsets, axis=0)
    noise = float(np.median(np.abs(offsets - shift))) * 1.4826
    tolerance = float(np.clip(3 * noise, 0.35, 1.25))
    keep = np.linalg.norm(offsets - shift, axis=1) <= tolerance
    if keep.sum() < 12 or keep.mean() < 0.55:
        return None, None
    shift = np.median(offsets[keep], axis=0)
    residuals = np.linalg.norm(offsets - shift, axis=1)
    keep = residuals <= tolerance
    if keep.sum() < 12 or keep.mean() < 0.55 or np.quantile(residuals[keep], 0.90) > 0.60:
        return None, None
    matrix = np.float64([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    return matrix, keep


def _resolve_settings(mode, min_similarity, max_drift):
    if mode == "auto":
        # A fixed safety ceiling limits automatic correction to small reframing.
        # Match noise later determines the required structural agreement.
        return 0.60, 0.08
    if mode != "manual":
        raise ValueError(f"Unknown pixel drift mode: {mode}")
    _validate_settings(min_similarity, max_drift)
    return min_similarity, max_drift


def estimate_alignment(reference, edited, min_similarity=0.60, max_drift=0.15,
                       mode="manual"):
    """Estimate edited -> reference homography on equal-sized uint8 grayscale arrays.

    Arrays are bounded previews; no output pixels are generated here. Expected
    nonmatches return a decision with no matrix. Invalid settings raise errors.
    """
    min_similarity, max_drift = _resolve_settings(mode, min_similarity, max_drift)
    if reference.ndim != 2 or reference.shape != edited.shape:
        raise ValueError("Alignment previews must be matching 2D grayscale arrays.")
    decision = AlignmentDecision()
    height, width = reference.shape
    if min(height, width) < 32:
        decision.reason = "image too small for reliable alignment"
        return decision
    cv2 = _opencv()
    detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    ref_keys, ref_desc = detector.detectAndCompute(reference, None)
    edit_keys, edit_desc = detector.detectAndCompute(edited, None)
    if ref_desc is None or edit_desc is None or min(len(ref_keys), len(edit_keys)) < 12:
        return decision
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_matches(a, b):
        matches = {}
        for pair in matcher.knnMatch(a, b, k=2):
            if len(pair) == 2:
                best, second = pair
                if best.distance < 0.72 * second.distance:
                    matches[best.queryIdx] = best.trainIdx
        return matches

    forward = ratio_matches(edit_desc, ref_desc)
    reverse = ratio_matches(ref_desc, edit_desc)
    pairs = [(a, b) for a, b in forward.items() if reverse.get(b) == a]
    if len(pairs) < 12:
        return decision
    edit_points = np.float32([edit_keys[a].pt for a, _ in pairs])
    ref_points = np.float32([ref_keys[b].pt for _, b in pairs])
    matrix, mask = None, None
    if mode == "auto":
        matrix, mask = _translation_fit(edit_points, ref_points)
        if matrix is not None:
            decision.model = "translation"
    if matrix is None:
        matrix, mask = cv2.findHomography(
            edit_points, ref_points, cv2.RANSAC, 2.5, maxIters=3000, confidence=0.995,
        )
    if matrix is None or mask is None:
        decision.reason = "no consistent transform"
        return decision
    keep = mask.ravel().astype(bool)
    decision.inliers = int(keep.sum())
    if decision.inliers < 12 or keep.mean() < 0.55:
        decision.reason = "too few consistent matches"
        return decision
    if not all(_spread(points[keep], width, height, cv2) for points in (edit_points, ref_points)):
        decision.reason = "matching content covers too little of the image"
        return decision
    if not _safe_transform(matrix, width, height, max_drift, cv2):
        decision.reason = "transform exceeds safe drift limits"
        return decision
    projected = cv2.perspectiveTransform(edit_points[keep][None], matrix)[0]
    residual = float(np.median(np.linalg.norm(projected - ref_points[keep], axis=1)))
    if residual > 1.5:
        decision.reason = "alignment residual is too large"
        return decision
    if mode == "auto":
        # Noisy matches and more flexible models need more independent evidence
        # across the image. Clean translations can tolerate larger local edits.
        min_similarity += min(0.10, residual * 0.10) + 0.05 * (1 - keep.mean())
        if decision.model == "perspective":
            min_similarity += 0.05
    valid = cv2.warpPerspective(np.ones_like(edited), matrix, (width, height), flags=cv2.INTER_NEAREST) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    warped = cv2.warpPerspective(edited, matrix, (width, height))
    after = _structure_scores(reference, warped, valid, cv2)
    decision.similarity = float((after >= 0.60).mean()) if len(after) else 0.0
    if len(after) < 32 or decision.similarity < min_similarity:
        decision.reason = "not enough shared image structure"
        return decision
    # Check a grid, not just matched keypoints, for negligible movement.
    grid = np.float32([(x, y) for y in np.linspace(0, height - 1, 5)
                      for x in np.linspace(0, width - 1, 5)])
    moved = cv2.perspectiveTransform(grid[None], matrix)[0]
    minimum_movement = 0.10 if mode == "auto" else 0.35
    if np.max(np.linalg.norm(moved - grid, axis=1)) < minimum_movement:
        decision.reason = "already aligned"
        return decision
    before = _structure_scores(reference, edited, valid, cv2)
    if float(after.mean()) < float(before.mean()) + 0.01:
        decision.reason = "correction does not improve shared structure"
        return decision
    decision.matrix = matrix / matrix[2, 2]
    decision.reason = "corrected"
    return decision


def _validate_settings(min_similarity, max_drift):
    if not math.isfinite(min_similarity) or not 0.30 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0.30 and 1.0.")
    if not math.isfinite(max_drift) or not 0.01 <= max_drift <= 0.30:
        raise ValueError("max_drift must be between 0.01 and 0.30.")


def _validate_image(image, name):
    import torch

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be an IMAGE tensor.")
    if image.ndim != 4 or min(image.shape) < 1 or image.shape[-1] not in (3, 4):
        raise ValueError(f"{name} must have nonempty shape [batch, height, width, 3 or 4].")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError(f"{name} must contain finite floating-point pixels.")


def _preview(image, size, cv2):
    import torch.nn.functional as functional

    # Reduce on the input device before transferring to CPU. Full-resolution
    # image data only crosses to CPU when an accepted transform needs a warp.
    rgb = functional.interpolate(
        image[..., :3].detach().movedim(-1, 0)[None].float(),
        size=size, mode="bilinear", align_corners=False, antialias=True,
    )[0].movedim(0, -1).clamp(0, 1).cpu().numpy()
    return cv2.cvtColor(np.rint(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def fix_pixel_drift(source_image, edited_image, min_similarity=0.60, max_drift=0.15,
                    mode="manual"):
    """Align each edit conservatively, retaining edit resolution/dtype/device.

    Singleton batches broadcast; other batch mismatches are errors. Rejected
    frames are copied exactly, without resizing, quantization or color transfer.
    """
    import torch

    _resolve_settings(mode, min_similarity, max_drift)
    _validate_image(source_image, "source_image")
    _validate_image(edited_image, "edited_image")
    source_count, edit_count = source_image.shape[0], edited_image.shape[0]
    if source_count != edit_count and min(source_count, edit_count) != 1:
        raise ValueError("Image batch sizes must match, or one must be a singleton.")
    cv2 = _opencv()
    height, width = edited_image.shape[1:3]
    scale = min(1.0, 1024 / max(height, width))
    preview_size = (max(1, round(height * scale)), max(1, round(width * scale)))
    ph, pw = preview_size
    # align_corners=False resize coordinates: preview = scale * (pixel + .5) - .5.
    to_preview = np.array([[pw / width, 0, (pw / width - 1) / 2],
                           [0, ph / height, (ph / height - 1) / 2], [0, 0, 1]])
    from_preview = np.linalg.inv(to_preview)
    previews = {}
    frames, messages = [], []
    any_corrected = False
    for index in range(max(source_count, edit_count)):
        source_index, edit_index = min(index, source_count - 1), min(index, edit_count - 1)
        for label, batch, frame_index in (("source", source_image, source_index), ("edit", edited_image, edit_index)):
            key = (label, frame_index)
            if key not in previews:
                previews[key] = _preview(batch[frame_index], preview_size, cv2)
        decision = estimate_alignment(previews["source", source_index], previews["edit", edit_index],
                                      min_similarity, max_drift, mode)
        frame = edited_image[edit_index]
        if decision.matrix is not None:
            native_matrix = from_preview @ decision.matrix @ to_preview
            # OpenCV resamples float pixels directly; detection alone uses uint8.
            warp_dtype = torch.float64 if frame.dtype == torch.float64 else torch.float32
            pixels = frame.detach().to(device="cpu", dtype=warp_dtype).numpy()
            pixels = cv2.warpPerspective(pixels, native_matrix, (width, height),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            frame = torch.from_numpy(pixels).to(device=edited_image.device, dtype=edited_image.dtype)
            any_corrected = True
        frames.append(frame)
        messages.append(
            f"Image {index + 1}: {decision.reason}; mode={mode}; model={decision.model}; "
            f"similarity={decision.similarity:.2f}; inliers={decision.inliers}"
        )
    result = torch.stack(frames) if any_corrected else edited_image.expand(len(frames), -1, -1, -1)
    return result, "\n".join(messages)
