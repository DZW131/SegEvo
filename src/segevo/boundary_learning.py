"""Boundary learning timeline and feature-separation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from segevo.artifacts import list_cases, list_epochs, load_array, sanitize_id
from segevo.feature_space import available_feature_layers
from segevo.metrics import boundary_dice, hd95, surface_dice


def boundary_learning_records(
    run_dir: str | Path,
    cases: Iterable[str] | None = None,
    layers: Iterable[str] | None = None,
    boundary_width: int = 2,
    surface_tolerance: float = 1.0,
) -> list[dict[str, object]]:
    """Return one boundary-learning record per case, epoch, and layer."""
    run_path = Path(run_dir)
    selected_cases = [sanitize_id(case_id) for case_id in cases] if cases else list_cases(run_path)
    selected_layers = list(layers) if layers is not None else available_feature_layers(run_path)
    if not selected_layers:
        selected_layers = ["metrics_only"]

    records: list[dict[str, object]] = []
    for case_id in selected_cases:
        case_path = run_path / "cases" / case_id
        gt_path = case_path / "gt.npy"
        if not gt_path.exists():
            continue
        gt = load_array(gt_path)
        for epoch in list_epochs(run_path, case_id):
            pred_path = case_path / "epochs" / f"{epoch:04d}" / "pred.npy"
            if not pred_path.exists():
                continue
            pred = load_array(pred_path)
            metric_values = {
                "boundary_dice": boundary_dice(pred, gt, width=boundary_width),
                "surface_dice": surface_dice(pred, gt, tolerance=surface_tolerance),
                "hd95": hd95(pred, gt),
            }
            feature_path = case_path / "epochs" / f"{epoch:04d}" / "features.npz"
            layer_values = _layer_separation_records(feature_path, selected_layers)
            for layer, separation in layer_values.items():
                records.append(
                    {
                        "case_id": case_id,
                        "epoch": int(epoch),
                        "layer": layer,
                        **metric_values,
                        **separation,
                    }
                )
    return records


def feature_boundary_separation(
    features: np.ndarray,
    region_ids: np.ndarray,
    region_names: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """Compute centroid distances for boundary-focused representation checks."""
    names = [str(name) for name in region_names]
    region_to_id = {name: index for index, name in enumerate(names)}
    boundary_features = _features_for_region(features, region_ids, region_to_id, "boundary")
    foreground_features = _features_for_region(features, region_ids, region_to_id, "foreground")
    hard_bg_features = _features_for_region(features, region_ids, region_to_id, "hard_background")

    boundary_within = _mean_centroid_distance(boundary_features)
    boundary_to_foreground = _centroid_distance(boundary_features, foreground_features)
    boundary_to_hard_bg = _centroid_distance(boundary_features, hard_bg_features)
    margin = _safe_ratio(boundary_to_hard_bg, boundary_within)

    return {
        "boundary_within_std": boundary_within,
        "boundary_to_foreground": boundary_to_foreground,
        "boundary_to_hard_background": boundary_to_hard_bg,
        "boundary_hard_margin": margin,
        "boundary_samples": float(len(boundary_features)),
        "hard_background_samples": float(len(hard_bg_features)),
    }


def _layer_separation_records(
    feature_path: Path,
    layers: list[str],
) -> dict[str, dict[str, float]]:
    empty = {layer: _empty_separation() for layer in layers}
    if not feature_path.exists():
        return empty

    with np.load(feature_path) as payload:
        if "feature_region_names" not in payload.files:
            return empty
        region_names = tuple(str(name) for name in payload["feature_region_names"].tolist())
        values: dict[str, dict[str, float]] = {}
        for layer in layers:
            sample_key = f"{layer}__samples"
            region_key = f"{layer}__sample_region_ids"
            if sample_key not in payload.files or region_key not in payload.files:
                values[layer] = _empty_separation()
                continue
            values[layer] = feature_boundary_separation(
                features=np.asarray(payload[sample_key], dtype=np.float32),
                region_ids=np.asarray(payload[region_key], dtype=np.int16),
                region_names=region_names,
            )
        return values


def _features_for_region(
    features: np.ndarray,
    region_ids: np.ndarray,
    region_to_id: dict[str, int],
    name: str,
) -> np.ndarray:
    channels = features.shape[1] if features.ndim == 2 else 0
    region_id = region_to_id.get(name)
    if region_id is None:
        return np.zeros((0, channels), dtype=np.float32)
    return np.asarray(features[region_ids == region_id], dtype=np.float32)


def _centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a.mean(axis=0) - b.mean(axis=0)
    return float(np.linalg.norm(diff) / np.sqrt(max(diff.size, 1)))


def _mean_centroid_distance(features: np.ndarray) -> float:
    if features.size == 0:
        return float("nan")
    centroid = features.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(features - centroid, axis=1)
    return float(distances.mean() / np.sqrt(max(features.shape[1], 1)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _empty_separation() -> dict[str, float]:
    return {
        "boundary_within_std": float("nan"),
        "boundary_to_foreground": float("nan"),
        "boundary_to_hard_background": float("nan"),
        "boundary_hard_margin": float("nan"),
        "boundary_samples": 0.0,
        "hard_background_samples": 0.0,
    }
