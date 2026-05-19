"""Feature-space loading and stable PCA projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from segevo.artifacts import list_cases, list_epochs, sanitize_id


@dataclass(frozen=True)
class FeatureSpace:
    layer: str
    features: np.ndarray
    region_ids: np.ndarray
    region_names: tuple[str, ...]
    case_ids: np.ndarray
    epochs: np.ndarray
    coords: np.ndarray


@dataclass(frozen=True)
class FeatureProjection:
    layer: str
    x: np.ndarray
    y: np.ndarray
    region_ids: np.ndarray
    region_names: tuple[str, ...]
    case_ids: np.ndarray
    epochs: np.ndarray
    feature_coords: np.ndarray
    explained_variance_ratio: tuple[float, float]


def available_feature_layers(run_dir: str | Path) -> list[str]:
    layers: set[str] = set()
    run_path = Path(run_dir)
    for case_id in list_cases(run_path):
        for epoch in list_epochs(run_path, case_id):
            features_path = _features_path(run_path, case_id, epoch)
            if not features_path.exists():
                continue
            with np.load(features_path) as payload:
                for name in payload.files:
                    if name.endswith("__samples"):
                        layers.add(name.removesuffix("__samples"))
    return sorted(layers)


def load_feature_space(
    run_dir: str | Path,
    layer: str,
    cases: list[str] | tuple[str, ...] | None = None,
    epochs: list[int] | tuple[int, ...] | None = None,
    max_points: int | None = None,
    seed: int = 13,
) -> FeatureSpace:
    run_path = Path(run_dir)
    selected_cases = [sanitize_id(case_id) for case_id in cases] if cases else list_cases(run_path)
    selected_epochs = set(int(epoch) for epoch in epochs) if epochs is not None else None

    features_by_file: list[np.ndarray] = []
    region_ids_by_file: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    epoch_ids: list[np.ndarray] = []
    coords_by_file: list[np.ndarray] = []
    region_names: tuple[str, ...] = ()

    for case_id in selected_cases:
        case_epochs = list_epochs(run_path, case_id)
        for epoch in case_epochs:
            if selected_epochs is not None and epoch not in selected_epochs:
                continue
            features_path = _features_path(run_path, case_id, epoch)
            if not features_path.exists():
                continue
            with np.load(features_path) as payload:
                sample_key = f"{layer}__samples"
                region_key = f"{layer}__sample_region_ids"
                coord_key = f"{layer}__sample_coords"
                if sample_key not in payload.files or region_key not in payload.files:
                    continue
                features = np.asarray(payload[sample_key], dtype=np.float32)
                region_ids = np.asarray(payload[region_key], dtype=np.int16)
                coords = _load_coords(payload, coord_key, features.shape[0])
                if features.size == 0 or region_ids.size == 0:
                    continue
                if not region_names and "feature_region_names" in payload.files:
                    names = payload["feature_region_names"].tolist()
                    region_names = tuple(str(name) for name in names)

            features_by_file.append(features)
            region_ids_by_file.append(region_ids)
            coords_by_file.append(coords)
            case_ids.append(np.asarray([case_id] * features.shape[0], dtype=object))
            epoch_ids.append(np.full(features.shape[0], int(epoch), dtype=np.int32))

    if not features_by_file:
        return FeatureSpace(
            layer=layer,
            features=np.zeros((0, 0), dtype=np.float32),
            region_ids=np.zeros((0,), dtype=np.int16),
            region_names=region_names,
            case_ids=np.zeros((0,), dtype=object),
            epochs=np.zeros((0,), dtype=np.int32),
            coords=np.zeros((0, 0), dtype=np.int32),
        )

    feature_space = FeatureSpace(
        layer=layer,
        features=np.concatenate(features_by_file, axis=0),
        region_ids=np.concatenate(region_ids_by_file, axis=0),
        region_names=region_names,
        case_ids=np.concatenate(case_ids, axis=0),
        epochs=np.concatenate(epoch_ids, axis=0),
        coords=np.concatenate(coords_by_file, axis=0),
    )
    if max_points is not None and feature_space.features.shape[0] > max_points:
        return downsample_feature_space(feature_space, max_points=max_points, seed=seed)
    return feature_space


def project_feature_space(feature_space: FeatureSpace) -> FeatureProjection:
    x, y, explained = pca_2d(feature_space.features)
    return FeatureProjection(
        layer=feature_space.layer,
        x=x,
        y=y,
        region_ids=feature_space.region_ids,
        region_names=feature_space.region_names,
        case_ids=feature_space.case_ids,
        epochs=feature_space.epochs,
        feature_coords=feature_space.coords,
        explained_variance_ratio=explained,
    )


def downsample_feature_space(
    feature_space: FeatureSpace,
    max_points: int,
    seed: int = 13,
) -> FeatureSpace:
    if feature_space.features.shape[0] <= max_points:
        return feature_space
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(feature_space.features.shape[0], size=max_points, replace=False))
    return FeatureSpace(
        layer=feature_space.layer,
        features=feature_space.features[keep],
        region_ids=feature_space.region_ids[keep],
        region_names=feature_space.region_names,
        case_ids=feature_space.case_ids[keep],
        epochs=feature_space.epochs[keep],
        coords=feature_space.coords[keep],
    )


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    if features.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, (0.0, 0.0)

    x = np.asarray(features, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}")
    if x.shape[0] == 1:
        zeros = np.zeros((1,), dtype=np.float32)
        return zeros, zeros, (0.0, 0.0)

    centered = x - x.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        projected_x = centered[:, 0].astype(np.float32)
        projected_y = np.zeros_like(projected_x)
        return projected_x, projected_y, (1.0, 0.0)

    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=np.float32)])

    explained = singular_values**2
    total = float(explained.sum())
    ratios = explained / total if total > 0 else np.zeros_like(explained)
    ratio_1 = float(ratios[0]) if ratios.size > 0 else 0.0
    ratio_2 = float(ratios[1]) if ratios.size > 1 else 0.0
    return coords[:, 0].astype(np.float32), coords[:, 1].astype(np.float32), (ratio_1, ratio_2)


def _features_path(run_dir: Path, case_id: str, epoch: int) -> Path:
    return run_dir / "cases" / sanitize_id(case_id) / "epochs" / f"{epoch:04d}" / "features.npz"


def _load_coords(payload: np.lib.npyio.NpzFile, key: str, count: int) -> np.ndarray:
    if key in payload.files:
        return np.asarray(payload[key], dtype=np.int32)
    return np.zeros((count, 0), dtype=np.int32)
