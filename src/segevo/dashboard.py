"""Streamlit dashboard for SegEvo artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from segevo.artifacts import list_cases, list_epochs, load_array, read_manifest
from segevo.boundary_learning import boundary_learning_records
from segevo.feature_space import (
    available_feature_layers,
    load_feature_space,
    project_feature_space,
)


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

    timeline_tab, feature_space_tab, boundary_tab = st.tabs(
        ["Case Timeline", "Feature Space", "Boundary Learning"]
    )

    with timeline_tab:
        _render_case_timeline(st, px, run_path, case_id, epoch, metrics)

    with feature_space_tab:
        _render_feature_space(st, px, run_path, case_id)

    with boundary_tab:
        _render_boundary_learning(st, px, run_path, case_id)


def _render_case_timeline(
    st: object,
    px: object,
    run_path: Path,
    case_id: str,
    epoch: int,
    metrics: pd.DataFrame,
) -> None:
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
            _plotly_chart(st, fig)

    features_path = epoch_path / "features.npz"
    if features_path.exists():
        st.subheader("Feature Summaries")
        with np.load(features_path) as features:
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
                _dataframe(st, pd.DataFrame(rows))

            sample_rows = _feature_sample_counts(features)
            if sample_rows:
                st.subheader("Feature Sample Counts")
                _dataframe(st, pd.DataFrame(sample_rows))


def _render_feature_space(st: object, px: object, run_path: Path, current_case_id: str) -> None:
    layers = available_feature_layers(run_path)
    if not layers:
        st.info("No feature samples found. Run a logger with attached layers first.")
        return

    controls, plot_area = st.columns([1, 3])
    with controls:
        layer = st.selectbox("Layer", layers)
        case_scope = st.radio("Cases", ["Current case", "All cases"], horizontal=False)
        if case_scope == "Current case":
            selected_cases = [current_case_id]
        else:
            selected_cases = list_cases(run_path)

        available_epochs = sorted(
            {
                epoch
                for selected_case in selected_cases
                for epoch in list_epochs(run_path, selected_case)
            }
        )
        default_epochs = available_epochs[-min(6, len(available_epochs)) :]
        selected_epochs = st.multiselect("Epochs", available_epochs, default=default_epochs)
        max_points = st.slider("Max points", min_value=500, max_value=10000, value=4000, step=500)

    if not selected_epochs:
        st.info("Select at least one epoch.")
        return

    space = load_feature_space(
        run_path,
        layer=layer,
        cases=selected_cases,
        epochs=selected_epochs,
        max_points=max_points,
    )
    if space.features.size == 0:
        st.info("No sampled features found for this layer/case/epoch selection.")
        return

    projection = project_feature_space(space)
    df = _projection_dataframe(projection)
    title = (
        f"{layer} 3D PCA "
        f"(PC1 {projection.explained_variance_ratio[0]:.1%}, "
        f"PC2 {projection.explained_variance_ratio[1]:.1%}, "
        f"PC3 {projection.explained_variance_ratio[2]:.1%})"
    )
    fig = px.scatter_3d(
        df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="region",
        symbol="epoch_label",
        hover_data=["case_id", "epoch", "coord"],
        title=title,
        opacity=0.72,
    )
    fig.update_traces(marker={"size": 6})
    fig.update_layout(
        height=720,
        scene={
            "xaxis_title": "PC1",
            "yaxis_title": "PC2",
            "zaxis_title": "PC3",
            "aspectmode": "cube",
        },
        legend_title_text="region, epoch",
    )
    with plot_area:
        _plotly_chart(st, fig)
        summary = (
            df.groupby(["epoch", "region"])
            .size()
            .reset_index(name="samples")
            .sort_values(["epoch", "region"])
        )
        _dataframe(st, summary)


def _render_boundary_learning(st: object, px: object, run_path: Path, current_case_id: str) -> None:
    layers = available_feature_layers(run_path)
    controls, plot_area = st.columns([1, 3])
    with controls:
        case_scope = st.radio(
            "Boundary cases",
            ["Current case", "All cases"],
            horizontal=False,
            key="boundary_case_scope",
        )
        if case_scope == "Current case":
            selected_cases = [current_case_id]
        else:
            selected_cases = list_cases(run_path)

        if layers:
            layer = st.selectbox("Boundary layer", layers, key="boundary_layer")
            selected_layers = [layer]
        else:
            layer = "metrics_only"
            selected_layers = [layer]
            st.info("No feature samples found; showing boundary metrics only.")

        boundary_width = st.slider(
            "Boundary width",
            min_value=1,
            max_value=8,
            value=2,
            step=1,
        )
        surface_tolerance = st.slider(
            "Surface tolerance",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
        )

    records = boundary_learning_records(
        run_path,
        cases=selected_cases,
        layers=selected_layers,
        boundary_width=boundary_width,
        surface_tolerance=surface_tolerance,
    )
    if not records:
        st.info("No boundary-learning records found for this selection.")
        return

    df = pd.DataFrame(records)
    numeric_columns = [
        "boundary_dice",
        "surface_dice",
        "hd95",
        "boundary_to_foreground",
        "boundary_to_hard_background",
        "boundary_hard_margin",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if case_scope == "All cases":
        plot_df = (
            df.groupby(["epoch", "layer"], as_index=False)[numeric_columns]
            .mean(numeric_only=True)
            .sort_values("epoch")
        )
    else:
        plot_df = df.sort_values("epoch")

    with plot_area:
        st.subheader("Boundary Metrics")
        metric_choices = [
            metric
            for metric in ["boundary_dice", "surface_dice", "hd95"]
            if metric in plot_df.columns
        ]
        selected_metrics = st.multiselect(
            "Boundary metrics",
            metric_choices,
            default=metric_choices[:2],
        )
        if selected_metrics:
            metric_long = plot_df.melt(
                id_vars=["epoch"],
                value_vars=selected_metrics,
                var_name="metric",
                value_name="value",
            )
            fig = px.line(metric_long, x="epoch", y="value", color="metric", markers=True)
            _plotly_chart(st, fig)

        st.subheader("Boundary Feature Separation")
        separation_choices = [
            metric
            for metric in [
                "boundary_to_hard_background",
                "boundary_to_foreground",
                "boundary_hard_margin",
            ]
            if metric in plot_df.columns and plot_df[metric].notna().any()
        ]
        if separation_choices:
            separation_long = plot_df.melt(
                id_vars=["epoch"],
                value_vars=separation_choices,
                var_name="metric",
                value_name="value",
            ).dropna(subset=["value"])
            fig = px.line(separation_long, x="epoch", y="value", color="metric", markers=True)
            _plotly_chart(st, fig)
        else:
            st.info("This selection does not contain enough boundary/hard-background samples.")

        _dataframe(st, df)


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


def _plotly_chart(st: object, fig: object) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def _dataframe(st: object, df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def _projection_dataframe(projection: object) -> pd.DataFrame:
    region_names = list(projection.region_names)
    region_labels = []
    for region_id in projection.region_ids:
        if int(region_id) < len(region_names):
            region_labels.append(region_names[int(region_id)])
        else:
            region_labels.append(str(int(region_id)))

    coords = [
        ",".join(str(int(value)) for value in row)
        for row in np.asarray(projection.feature_coords)
    ]
    return pd.DataFrame(
        {
            "pc1": projection.x,
            "pc2": projection.y,
            "pc3": projection.z,
            "region": region_labels,
            "case_id": projection.case_ids,
            "epoch": projection.epochs,
            "epoch_label": projection.epochs.astype(str),
            "coord": coords,
        }
    )
