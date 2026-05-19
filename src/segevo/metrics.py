"""Segmentation metrics and error maps."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def binarize(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred_b = binarize(pred)
    gt_b = binarize(gt)
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(pred_b, gt_b).sum() + eps) / (denom + eps))


def volume_error(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred_count = float(binarize(pred).sum())
    gt_count = float(binarize(gt).sum())
    if gt_count == 0:
        return 0.0 if pred_count == 0 else float("inf")
    return float((pred_count - gt_count) / (gt_count + eps))


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_b = binarize(pred)
    gt_b = binarize(gt)
    error = np.zeros(pred_b.shape, dtype=np.uint8)
    error[np.logical_and(pred_b, gt_b)] = 1
    error[np.logical_and(pred_b, np.logical_not(gt_b))] = 2
    error[np.logical_and(np.logical_not(pred_b), gt_b)] = 3
    return error


def boundary(mask: np.ndarray) -> np.ndarray:
    mask_b = binarize(mask)
    if not mask_b.any():
        return np.zeros(mask_b.shape, dtype=bool)
    structure = ndimage.generate_binary_structure(mask_b.ndim, 1)
    eroded = ndimage.binary_erosion(mask_b, structure=structure, border_value=0)
    return np.logical_xor(mask_b, eroded)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...] | None = None) -> float:
    pred_b = binarize(pred)
    gt_b = binarize(gt)
    if not pred_b.any() and not gt_b.any():
        return 0.0
    if not pred_b.any() or not gt_b.any():
        return float("inf")

    pred_surface = boundary(pred_b)
    gt_surface = boundary(gt_b)
    pred_to_gt = _surface_distances(pred_surface, gt_surface, spacing)
    gt_to_pred = _surface_distances(gt_surface, pred_surface, spacing)
    distances = np.concatenate([pred_to_gt, gt_to_pred])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float = 1.0,
    spacing: tuple[float, ...] | None = None,
) -> float:
    pred_b = binarize(pred)
    gt_b = binarize(gt)
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0

    pred_surface = boundary(pred_b)
    gt_surface = boundary(gt_b)
    pred_to_gt = _surface_distances(pred_surface, gt_surface, spacing)
    gt_to_pred = _surface_distances(gt_surface, pred_surface, spacing)
    numerator = np.count_nonzero(pred_to_gt <= tolerance)
    numerator += np.count_nonzero(gt_to_pred <= tolerance)
    denominator = pred_to_gt.size + gt_to_pred.size
    return float(numerator / denominator) if denominator else 1.0


def summarize_binary_segmentation(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    surface_tolerance: float = 1.0,
) -> dict[str, float]:
    return {
        "dice": dice_score(pred, gt),
        "hd95": hd95(pred, gt, spacing=spacing),
        "surface_dice": surface_dice(
            pred,
            gt,
            tolerance=surface_tolerance,
            spacing=spacing,
        ),
        "volume_error": volume_error(pred, gt),
    }


def _surface_distances(
    source_surface: np.ndarray,
    target_surface: np.ndarray,
    spacing: tuple[float, ...] | None,
) -> np.ndarray:
    if not source_surface.any() or not target_surface.any():
        return np.asarray([], dtype=float)
    sampling = spacing if spacing is not None else None
    distance_map = ndimage.distance_transform_edt(np.logical_not(target_surface), sampling=sampling)
    return distance_map[source_surface]
