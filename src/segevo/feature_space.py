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
    spatial_shapes: np.ndarray | None = None


@dataclass(frozen=True)
class FeatureProjection:
    layer: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    region_ids: np.ndarray
    region_names: tuple[str, ...]
    case_ids: np.ndarray
    epochs: np.ndarray
    feature_coords: np.ndarray
    explained_variance_ratio: tuple[float, ...]
    feature_spatial_shapes: np.ndarray | None = None


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
    spatial_shapes_by_file: list[np.ndarray] = []
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
                shape_key = f"{layer}__sample_spatial_shape"
                if sample_key not in payload.files or region_key not in payload.files:
                    continue
                features = np.asarray(payload[sample_key], dtype=np.float32)
                region_ids = np.asarray(payload[region_key], dtype=np.int16)
                coords = _load_coords(payload, coord_key, features.shape[0])
                spatial_shape = _load_spatial_shape(payload, shape_key, coords)
                if features.size == 0 or region_ids.size == 0:
                    continue
                if not region_names and "feature_region_names" in payload.files:
                    names = payload["feature_region_names"].tolist()
                    region_names = tuple(str(name) for name in names)

            features_by_file.append(features)
            region_ids_by_file.append(region_ids)
            coords_by_file.append(coords)
            spatial_shapes_by_file.append(
                np.repeat(spatial_shape[None, :], features.shape[0], axis=0)
            )
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
            spatial_shapes=np.zeros((0, 0), dtype=np.int32),
        )

    feature_space = FeatureSpace(
        layer=layer,
        features=np.concatenate(features_by_file, axis=0),
        region_ids=np.concatenate(region_ids_by_file, axis=0),
        region_names=region_names,
        case_ids=np.concatenate(case_ids, axis=0),
        epochs=np.concatenate(epoch_ids, axis=0),
        coords=np.concatenate(coords_by_file, axis=0),
        spatial_shapes=np.concatenate(spatial_shapes_by_file, axis=0),
    )
    if max_points is not None and feature_space.features.shape[0] > max_points:
        return downsample_feature_space(feature_space, max_points=max_points, seed=seed)
    return feature_space


def project_feature_space(feature_space: FeatureSpace) -> FeatureProjection:
    x, y, z, explained = pca_3d(feature_space.features)
    return FeatureProjection(
        layer=feature_space.layer,
        x=x,
        y=y,
        z=z,
        region_ids=feature_space.region_ids,
        region_names=feature_space.region_names,
        case_ids=feature_space.case_ids,
        epochs=feature_space.epochs,
        feature_coords=feature_space.coords,
        explained_variance_ratio=explained,
        feature_spatial_shapes=feature_space.spatial_shapes,
    )


def downsample_feature_space(
    feature_space: FeatureSpace,
    max_points: int,
    seed: int = 13,
) -> FeatureSpace:
    if feature_space.features.shape[0] <= max_points:
        return feature_space
    rng = np.random.default_rng(seed)
    keep = _stratified_indices(
        epochs=feature_space.epochs,
        region_ids=feature_space.region_ids,
        max_points=max_points,
        rng=rng,
    )
    return FeatureSpace(
        layer=feature_space.layer,
        features=feature_space.features[keep],
        region_ids=feature_space.region_ids[keep],
        region_names=feature_space.region_names,
        case_ids=feature_space.case_ids[keep],
        epochs=feature_space.epochs[keep],
        coords=feature_space.coords[keep],
        spatial_shapes=(
            feature_space.spatial_shapes[keep]
            if feature_space.spatial_shapes is not None
            else None
        ),
    )


def _stratified_indices(
    epochs: np.ndarray,
    region_ids: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    keys = np.column_stack([np.asarray(epochs), np.asarray(region_ids)])
    groups: list[np.ndarray] = []
    for key in np.unique(keys, axis=0):
        group = np.flatnonzero(np.all(keys == key, axis=1))
        if group.size > 0:
            groups.append(group)

    if not groups:
        return np.zeros((0,), dtype=np.int64)

    base_quota = max(1, max_points // len(groups))
    selected: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []
    remaining = max_points

    for group in groups:
        if group.size <= base_quota:
            selected.append(group)
            remaining -= group.size
        else:
            take = rng.choice(group, size=base_quota, replace=False)
            selected.append(take)
            leftovers.append(np.setdiff1d(group, take, assume_unique=False))
            remaining -= base_quota

    if remaining > 0 and leftovers:
        pool = np.concatenate([leftover for leftover in leftovers if leftover.size > 0])
        if pool.size > 0:
            extra = rng.choice(pool, size=min(remaining, pool.size), replace=False)
            selected.append(extra)

    keep = np.concatenate(selected)
    if keep.size > max_points:
        keep = rng.choice(keep, size=max_points, replace=False)
    return np.sort(keep.astype(np.int64, copy=False))


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    coords, ratios = pca_nd(features, n_components=2)
    return coords[:, 0], coords[:, 1], (ratios[0], ratios[1])


def pca_3d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    coords, ratios = pca_nd(features, n_components=3)
    return coords[:, 0], coords[:, 1], coords[:, 2], (ratios[0], ratios[1], ratios[2])


def pca_nd(features: np.ndarray, n_components: int) -> tuple[np.ndarray, tuple[float, ...]]:
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    if features.size == 0:
        return np.zeros((0, n_components), dtype=np.float32), tuple(0.0 for _ in range(n_components))

    x = np.asarray(features, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}")
    if x.shape[0] == 1:
        return np.zeros((1, n_components), dtype=np.float32), tuple(
            0.0 for _ in range(n_components)
        )

    centered = x - x.mean(axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    usable_components = min(n_components, vt.shape[0])
    coords = np.zeros((centered.shape[0], n_components), dtype=np.float32)
    if usable_components > 0:
        components = vt[:usable_components]
        coords[:, :usable_components] = (centered @ components.T).astype(np.float32)

    explained = singular_values**2
    total = float(explained.sum())
    ratios = explained / total if total > 0 else np.zeros_like(explained)
    padded_ratios = [float(ratios[index]) if index < ratios.size else 0.0 for index in range(n_components)]
    return coords, tuple(padded_ratios)


def _features_path(run_dir: Path, case_id: str, epoch: int) -> Path:
    return run_dir / "cases" / sanitize_id(case_id) / "epochs" / f"{epoch:04d}" / "features.npz"


def _load_coords(payload: np.lib.npyio.NpzFile, key: str, count: int) -> np.ndarray:
    if key in payload.files:
        return np.asarray(payload[key], dtype=np.int32)
    return np.zeros((count, 0), dtype=np.int32)


def _load_spatial_shape(
    payload: np.lib.npyio.NpzFile,
    key: str,
    coords: np.ndarray,
) -> np.ndarray:
    if key in payload.files:
        shape = np.asarray(payload[key], dtype=np.int32).ravel()
        if shape.size > 0:
            return shape
    if coords.ndim != 2:
        return np.zeros((0,), dtype=np.int32)
    if coords.size == 0:
        return np.zeros((coords.shape[1],), dtype=np.int32)
    return np.maximum(coords.max(axis=0) + 1, 1).astype(np.int32, copy=False)
