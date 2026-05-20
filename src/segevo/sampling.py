"""Feature-region sampling for segmentation representation analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from segevo.metrics import binarize, boundary

REGION_NAMES = (
    "foreground",
    "boundary",
    "hard_background",
    "false_positive",
    "false_negative",
)


@dataclass(frozen=True)
class FeatureSamples:
    features: np.ndarray
    region_ids: np.ndarray
    coords: np.ndarray
    spatial_shape: np.ndarray


def sample_feature_regions(
    feature: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    max_samples_per_region: int = 128,
    boundary_width: int = 2,
    hard_background_width: int = 8,
    seed: int | None = None,
) -> FeatureSamples | None:
    """Sample feature vectors from segmentation-relevant regions.

    ``feature`` is expected to be channel-first, e.g. ``[C, H, W]`` or
    ``[B, C, H, W]`` from a PyTorch hook. Only the first batch item is sampled.
    """
    feature_cf = as_channel_first_feature(feature)
    if feature_cf is None:
        return None

    spatial_shape = feature_cf.shape[1:]
    regions = build_region_masks(
        gt=gt,
        pred=pred,
        target_shape=spatial_shape,
        boundary_width=boundary_width,
        hard_background_width=hard_background_width,
    )
    rng = np.random.default_rng(seed)
    sampled_features: list[np.ndarray] = []
    sampled_region_ids: list[np.ndarray] = []
    sampled_coords: list[np.ndarray] = []

    for region_id, region_name in enumerate(REGION_NAMES):
        coords = np.argwhere(regions[region_name])
        if coords.size == 0:
            continue
        if coords.shape[0] > max_samples_per_region:
            take = rng.choice(coords.shape[0], size=max_samples_per_region, replace=False)
            coords = coords[np.sort(take)]

        vectors = feature_cf[(slice(None),) + tuple(coords.T)].T.astype(np.float32, copy=False)
        sampled_features.append(vectors)
        sampled_region_ids.append(np.full(coords.shape[0], region_id, dtype=np.int16))
        sampled_coords.append(coords.astype(np.int32, copy=False))

    if not sampled_features:
        return FeatureSamples(
            features=np.zeros((0, feature_cf.shape[0]), dtype=np.float32),
            region_ids=np.zeros((0,), dtype=np.int16),
            coords=np.zeros((0, len(spatial_shape)), dtype=np.int32),
            spatial_shape=np.asarray(spatial_shape, dtype=np.int32),
        )

    return FeatureSamples(
        features=np.concatenate(sampled_features, axis=0),
        region_ids=np.concatenate(sampled_region_ids, axis=0),
        coords=np.concatenate(sampled_coords, axis=0),
        spatial_shape=np.asarray(spatial_shape, dtype=np.int32),
    )


def as_channel_first_feature(feature: np.ndarray) -> np.ndarray | None:
    array = np.asarray(feature)
    if array.ndim < 3:
        return None
    if array.ndim >= 4:
        array = array[0]
    if array.ndim < 3:
        return None
    return np.asarray(array, dtype=np.float32)


def build_region_masks(
    gt: np.ndarray,
    pred: np.ndarray,
    target_shape: tuple[int, ...],
    boundary_width: int = 2,
    hard_background_width: int = 8,
) -> dict[str, np.ndarray]:
    gt_b = binarize(gt)
    pred_b = binarize(pred)
    if gt_b.shape != pred_b.shape:
        raise ValueError(
            f"gt and pred must have the same shape, got {gt_b.shape} and {pred_b.shape}"
        )

    structure = ndimage.generate_binary_structure(gt_b.ndim, 1)
    gt_boundary = boundary(gt_b)
    boundary_band = ndimage.binary_dilation(
        gt_boundary,
        structure=structure,
        iterations=max(1, int(boundary_width)),
    )
    near_foreground = ndimage.binary_dilation(
        gt_b,
        structure=structure,
        iterations=max(1, int(hard_background_width)),
    )

    false_positive = np.logical_and(pred_b, np.logical_not(gt_b))
    false_negative = np.logical_and(np.logical_not(pred_b), gt_b)
    boundary_region = boundary_band & ~(false_positive | false_negative)
    foreground = gt_b & ~boundary_band & ~false_negative
    hard_background = near_foreground & ~gt_b & ~false_positive

    source_regions = {
        "foreground": foreground,
        "boundary": boundary_region,
        "hard_background": hard_background,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
    return {
        name: project_mask_to_shape(mask, target_shape)
        for name, mask in source_regions.items()
    }


def project_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    mask_b = binarize(mask)
    if mask_b.ndim != len(target_shape):
        raise ValueError(f"mask ndim {mask_b.ndim} does not match target shape {target_shape}")
    projected = np.zeros(target_shape, dtype=bool)
    coords = np.argwhere(mask_b)
    if coords.size == 0:
        return projected

    source_shape = np.asarray(mask_b.shape, dtype=np.float64)
    target = np.asarray(target_shape, dtype=np.float64)
    target_coords = np.floor((coords + 0.5) * target / source_shape).astype(np.int64)
    for axis, size in enumerate(target_shape):
        target_coords[:, axis] = np.clip(target_coords[:, axis], 0, size - 1)
    projected[tuple(target_coords.T)] = True
    return projected
