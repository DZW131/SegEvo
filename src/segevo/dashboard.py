"""Streamlit dashboard for SegEvo artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from segevo.artifacts import list_cases, list_epochs, load_array, read_manifest


ERROR_COLORS = np.asarray(
    [
        [0, 0, 0],
        [52, 199, 89],
        [255, 149, 0],
        [255, 69, 58],
    ],
    dtype=np.float32,
) / 255.0


def run_dashboard(run_dir: str | Path) -> None:
    import plotly.express as px
    import streamlit as st

    run_path = Path(run_dir)
    manifest = read_manifest(run_path)
    st.set_page_config(page_title="SegEvo", layout="wide")

    st.title("SegEvo")
    st.caption(manifest.get("project", "Training-process observability for segmentation"))

    cases = list_cases(run_path)
    if not cases:
        st.warning("No cases found. Generate a demo with `segevo-demo --out runs/demo`.")
        return

    metrics_path = run_path / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    with st.sidebar:
        st.header("Run")
        st.write(f"`{run_path}`")
        case_id = st.selectbox("Case", cases)
        epochs = list_epochs(run_path, case_id)
        if not epochs:
            st.warning("This case has no epoch artifacts.")
            return
        epoch = st.select_slider("Epoch", options=epochs, value=epochs[-1])

    case_path = run_path / "cases" / case_id
    epoch_path = case_path / "epochs" / f"{epoch:04d}"
    image = load_array(case_path / "image.npy")
    gt = load_array(case_path / "gt.npy")
    pred = load_array(epoch_path / "pred.npy")
    err = load_array(epoch_path / "error.npy")

    image_slice, gt_slice, pred_slice, err_slice = _select_slice(st, image, gt, pred, err)

    col_image, col_pred, col_error = st.columns(3)
    with col_image:
        st.subheader("Image + GT")
        st.image(_overlay(image_slice, gt_slice, color=(52, 199, 89)), clamp=True)
    with col_pred:
        st.subheader("Image + Prediction")
        st.image(_overlay(image_slice, pred_slice, color=(0, 122, 255)), clamp=True)
    with col_error:
        st.subheader("Error Map")
        st.image(_error_overlay(image_slice, err_slice), clamp=True)

    if not metrics.empty:
        st.subheader("Metrics Timeline")
        case_metrics = metrics[metrics["case_id"].astype(str) == str(case_id)]
        if case_metrics.empty:
            case_metrics = metrics
        metric_names = [
            name
            for name in ["dice", "surface_dice", "hd95", "volume_error"]
            if name in case_metrics.columns
        ]
        selected = st.multiselect("Metrics", metric_names, default=metric_names[:2])
        if selected:
            long_df = case_metrics.melt(
                id_vars=["epoch"],
                value_vars=selected,
                var_name="metric",
                value_name="value",
            )
            fig = px.line(long_df, x="epoch", y="value", color="metric", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    features_path = epoch_path / "features.npz"
    if features_path.exists():
        st.subheader("Feature Summaries")
        features = np.load(features_path)
        rows = []
        for name in features.files:
            if not name.endswith("__summary"):
                continue
            values = np.asarray(features[name]).ravel()
            rows.append(
                {
                    "layer": name.replace("__summary", ""),
                    "mean": values[0] if values.size > 0 else np.nan,
                    "std": values[1] if values.size > 1 else np.nan,
                    "min": values[2] if values.size > 2 else np.nan,
                    "max": values[3] if values.size > 3 else np.nan,
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        sample_rows = _feature_sample_counts(features)
        if sample_rows:
            st.subheader("Feature Sample Counts")
            st.dataframe(pd.DataFrame(sample_rows), use_container_width=True)


def _select_slice(st: object, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    image = arrays[0]
    if image.ndim <= 2:
        return tuple(np.asarray(array) for array in arrays)

    axis = st.sidebar.selectbox("Slice axis", list(range(image.ndim)), index=0)
    max_index = image.shape[axis] - 1
    index = st.sidebar.slider("Slice", min_value=0, max_value=max_index, value=max_index // 2)
    return tuple(
        np.take(array, index, axis=axis) if array.ndim == image.ndim else array
        for array in arrays
    )


def _normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1)


def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.38,
) -> np.ndarray:
    base = np.repeat(_normalize(image)[..., None], 3, axis=-1)
    mask_b = np.asarray(mask) > 0
    overlay_color = np.asarray(color, dtype=np.float32) / 255.0
    base[mask_b] = (1.0 - alpha) * base[mask_b] + alpha * overlay_color
    return base


def _error_overlay(image: np.ndarray, err: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    base = np.repeat(_normalize(image)[..., None], 3, axis=-1)
    err_i = np.asarray(err, dtype=np.int64)
    mask = err_i > 0
    colors = ERROR_COLORS[np.clip(err_i, 0, len(ERROR_COLORS) - 1)]
    base[mask] = (1.0 - alpha) * base[mask] + alpha * colors[mask]
    return base


def _feature_sample_counts(features: np.lib.npyio.NpzFile) -> list[dict[str, object]]:
    if "feature_region_names" in features.files:
        region_names = [str(name) for name in features["feature_region_names"].tolist()]
    else:
        region_names = []
    rows: list[dict[str, object]] = []
    for name in features.files:
        if not name.endswith("__sample_region_ids"):
            continue
        layer = name.replace("__sample_region_ids", "")
        region_ids = np.asarray(features[name], dtype=np.int64)
        for region_id, count in zip(*np.unique(region_ids, return_counts=True)):
            if region_id < len(region_names):
                region_name = region_names[region_id]
            else:
                region_name = str(region_id)
            rows.append({"layer": layer, "region": region_name, "samples": int(count)})
    return rows
