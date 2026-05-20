"""Streamlit dashboard for SegEvo artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from segevo.artifacts import list_cases, list_epochs, load_array, read_manifest, sanitize_id
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

REGION_COLORS = {
    "foreground": "#0074D9",
    "boundary": "#7FDBFF",
    "hard_background": "#FF4136",
    "false_positive": "#FF851B",
    "false_negative": "#2ECC40",
}

CORE_SURFACE_REGIONS = ("foreground", "boundary", "hard_background")
ERROR_REGIONS = ("false_positive", "false_negative")
REGION_PRESETS = {
    "All regions": tuple(REGION_COLORS),
    "Core regions": CORE_SURFACE_REGIONS,
    "Errors only (FP/FN)": ERROR_REGIONS,
    "Boundary focus": ("boundary", "hard_background", "false_positive", "false_negative"),
}

SEPARATION_LABELS = {
    "boundary_within": "boundary spread",
    "boundary_to_foreground": "boundary -> foreground",
    "boundary_to_hard_background": "boundary -> hard bg",
    "fp_to_foreground": "FP -> foreground",
    "fn_to_foreground": "FN -> foreground",
    "boundary_hard_margin": "boundary hard margin",
}

PAGE_GUIDES = {
    "case_timeline": """
**English**

- Use `Case` and `Epoch` in the sidebar to replay one fixed probe case through training.
- `Image + GT` overlays the ground-truth mask in green; `Image + Prediction` overlays the
  model prediction in blue.
- `Error Map` uses green for true positives, orange for false positives, and red for false
  negatives.
- Metric curves show whether this case is improving. Dice and surface Dice are better when
  higher; HD95 is better when lower; volume error is better when closer to zero.
- Feature summaries and sample counts confirm which network layers were logged at this epoch.

**中文**

- 在左侧用 `Case` 和 `Epoch` 回放同一个固定 probe 病例在训练过程中的变化。
- `Image + GT` 用绿色叠加真实标注；`Image + Prediction` 用蓝色叠加模型预测。
- `Error Map` 中绿色是真阳性，橙色是假阳性，红色是假阴性。
- 指标曲线用于判断这个病例是否在变好：Dice 和 surface Dice 越高越好，HD95 越低越好，
  volume error 越接近 0 越好。
- Feature summaries 和 sample counts 用来确认当前 epoch 记录了哪些网络层特征。
""",
    "feature_space": """
**English**

- This page projects sampled high-dimensional layer features into a stable 3D PCA space.
  The PC axes are relative coordinates, so focus on clustering, distance, and trajectories.
- `Layer` chooses the hooked network layer. Deeper layers usually carry more semantic
  information; shallower layers often reflect texture and color.
- `Epoch` is synced with the sidebar epoch so every tab is looking at the same training
  moment. Enable `Epoch playback` to animate sampled points up to that epoch.
- `Region preset` quickly switches between all regions, core regions, FP/FN errors, and
  boundary-focused views; `Regions` still lets you fine tune the visible groups.
- `Surface` can add convex hulls or density clouds for foreground, boundary, and hard
  background clusters.
- Click a point to inspect its case, epoch, region, feature coordinate, and mapped image
  location. `Export 3D HTML` saves a shareable interactive plot.
- `Separation` summarizes distances between region centroids. Larger boundary-to-hard
  background distance often means the model separates boundary from confusing background
  more clearly.

**中文**

- 本页把采样到的高维中间层特征投影到稳定的 3D PCA 空间。PC 轴是相对坐标，
  重点看点群是否分开、距离是否变大、中心轨迹如何移动。
- `Layer` 选择被 hook 的网络层。深层通常更偏语义，浅层通常更偏颜色、纹理和边缘。
- `Epoch` 跟左侧全局 epoch 同步，确保三个页面看的都是同一个训练时刻。打开
  `Epoch playback` 可以播放到当前 epoch 为止的点云变化。
- `Region preset` 可以快速切换全部区域、核心区域、只看 FP/FN、边界聚焦视图；
  `Regions` 仍然可以手动细调显示哪些点群。
- `Surface` 可以给 foreground、boundary、hard background 加 convex hull 或
  density cloud，帮助看 3D 点群边界和覆盖范围。
- 点击任意点可以查看它来自哪个 case、epoch、region、feature 坐标，以及映射回原图的
  位置。`Export 3D HTML` 可以导出可分享的交互式 3D 图。
- `Separation` 是区域中心距离的定量摘要。boundary 到 hard background 距离越大，
  通常说明模型越能把边界和易混背景区分开。
""",
    "boundary_learning": """
**English**

- This page focuses on boundary learning rather than whole-mask overlap.
- `Boundary width` controls how thick the boundary band is when computing boundary Dice.
- `Surface tolerance` controls the accepted distance for surface Dice.
- Boundary Dice and surface Dice are better when higher; HD95 is better when lower.
- Boundary feature separation compares boundary features with foreground and hard
  background features. A rising separation trend usually means boundary representations
  are becoming more stable.

**中文**

- 本页专门看边界学习，而不只是整体 mask 重叠程度。
- `Boundary width` 控制计算 boundary Dice 时边界带的宽度。
- `Surface tolerance` 控制 surface Dice 允许的边界距离误差。
- boundary Dice 和 surface Dice 越高越好；HD95 越低越好。
- Boundary feature separation 比较边界特征与 foreground / hard background 特征。
  如果分离趋势上升，通常说明模型对边界的表征更稳定了。
""",
}


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
        if len(epochs) == 1:
            epoch = epochs[0]
            st.write(f"Epoch: `{epoch}`")
        else:
            epoch = st.select_slider("Epoch", options=epochs, value=epochs[-1])

    timeline_tab, feature_space_tab, boundary_tab = st.tabs(
        ["Case Timeline", "Feature Space", "Boundary Learning"]
    )

    with timeline_tab:
        _render_case_timeline(st, px, run_path, case_id, epoch, metrics)

    with feature_space_tab:
        _render_feature_space(st, px, run_path, case_id, epoch)

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
    _render_page_guide(st, "case_timeline")

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


def _render_feature_space(
    st: object,
    px: object,
    run_path: Path,
    current_case_id: str,
    sidebar_epoch: int,
) -> None:
    import plotly.graph_objects as go

    _render_page_guide(st, "feature_space")

    layers = available_feature_layers(run_path)
    if not layers:
        st.info("No feature samples found. Run a logger with attached layers first.")
        return

    controls, plot_area, metrics_area = st.columns([1.1, 3.2, 1.15])
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
        if not available_epochs:
            st.info("No epochs found for this case selection.")
            return
        focus_epoch = _nearest_epoch(available_epochs, sidebar_epoch)
        if focus_epoch == sidebar_epoch:
            st.write(f"Epoch: `{focus_epoch}`")
        else:
            st.write(
                f"Epoch: `{focus_epoch}` "
                f"(nearest available to sidebar `{sidebar_epoch}`)"
            )
        st.caption("Synced with the sidebar epoch.")
        display_mode = st.selectbox(
            "Display",
            ["Points + centroid trajectories", "Points only", "Centroids only"],
        )
        max_points = st.slider("Max points", min_value=500, max_value=10000, value=4000, step=500)

    space = load_feature_space(
        run_path,
        layer=layer,
        cases=selected_cases,
        epochs=available_epochs,
        max_points=max_points,
    )
    if space.features.size == 0:
        st.info("No sampled features found for this layer/case/epoch selection.")
        return

    projection = project_feature_space(space)
    df = _projection_dataframe(projection)
    all_regions = _ordered_regions(df["region"].dropna().unique())

    with controls:
        region_preset = st.selectbox("Region preset", list(REGION_PRESETS))
        default_regions = _default_regions_for_preset(all_regions, region_preset)
        selected_regions = st.multiselect(
            "Regions",
            all_regions,
            default=default_regions,
            key=f"feature_regions_{region_preset}",
        )
        animate_epochs = st.checkbox(
            "Epoch playback",
            value=False,
            help="Animate sampled points across epochs up to the selected epoch.",
        )
        surface_mode = st.selectbox("Surface", ["None", "Convex hull", "Density cloud"])
        if surface_mode == "None":
            surface_regions: list[str] = []
        else:
            surface_region_options = [
                region for region in CORE_SURFACE_REGIONS if region in all_regions
            ]
            surface_regions = st.multiselect(
                "Surface regions",
                all_regions,
                default=surface_region_options,
            )

    if not selected_regions:
        st.info("Select at least one region.")
        return

    history_df = df[(df["epoch"] <= focus_epoch) & df["region"].isin(selected_regions)].copy()
    points_df = history_df[history_df["epoch"] == focus_epoch].copy()
    centroid_df = _centroid_dataframe(history_df)

    if history_df.empty:
        st.info("No sampled features found for this epoch/region selection.")
        return

    title = (
        f"{layer} 3D PCA "
        f"(PC1 {projection.explained_variance_ratio[0]:.1%}, "
        f"PC2 {projection.explained_variance_ratio[1]:.1%}, "
        f"PC3 {projection.explained_variance_ratio[2]:.1%})"
    )
    show_points = display_mode in {"Points + centroid trajectories", "Points only"}
    show_centroids = display_mode in {"Points + centroid trajectories", "Centroids only"}
    plot_points_df = history_df if animate_epochs else points_df

    if show_points and not plot_points_df.empty:
        scatter_kwargs = {
            "data_frame": plot_points_df,
            "x": "pc1",
            "y": "pc2",
            "z": "pc3",
            "color": "region",
            "color_discrete_map": REGION_COLORS,
            "category_orders": {"region": all_regions},
            "hover_data": {
                "case_id": True,
                "epoch": True,
                "feature_coord": True,
                "spatial_shape": True,
                "point_id": False,
            },
            "custom_data": [
                "point_id",
                "case_id",
                "epoch",
                "region",
                "feature_coord",
                "spatial_shape",
            ],
            "title": title,
            "opacity": 0.7,
        }
        if animate_epochs:
            scatter_kwargs["animation_frame"] = "epoch_label"
            scatter_kwargs["animation_group"] = "point_id"
        fig = px.scatter_3d(
            **scatter_kwargs,
        )
        fig.update_traces(marker={"size": 5})
    else:
        fig = go.Figure()
        fig.update_layout(title=title)

    if surface_mode != "None":
        _add_region_surface_traces(fig, go, points_df, surface_regions, surface_mode)

    if show_centroids:
        _add_centroid_traces(fig, go, centroid_df, selected_regions, show_legend=not show_points)

    fig.update_layout(
        height=720,
        scene={
            "xaxis_title": "PC1",
            "yaxis_title": "PC2",
            "zaxis_title": "PC3",
            "aspectmode": "cube",
        },
        legend_title_text="region",
    )
    with plot_area:
        selection_event = _plotly_chart(
            st,
            fig,
            key="feature_space_3d",
            selection_mode="points" if show_points else None,
        )
        st.download_button(
            "Export 3D HTML",
            data=fig.to_html(include_plotlyjs="cdn", full_html=True),
            file_name=_feature_space_html_name(layer, focus_epoch, region_preset),
            mime="text/html",
        )
        summary = (
            history_df.groupby(["epoch", "region"])
            .size()
            .reset_index(name="samples")
            .sort_values(["epoch", "region"])
        )
        _dataframe(st, summary)

    with metrics_area:
        st.subheader("Point Detail")
        selected_point_id = _selected_point_id(selection_event)
        if selected_point_id is None:
            st.info("Click a point in the 3D plot to inspect its source location.")
        else:
            selected_rows = df[df["point_id"] == selected_point_id]
            if selected_rows.empty:
                st.info("Selected point is no longer visible after filtering.")
            else:
                _render_feature_point_detail(st, run_path, selected_rows.iloc[0])

        st.subheader("Separation")
        metrics = _feature_separation_metrics(history_df, focus_epoch)
        if metrics.empty:
            st.info("Need boundary/foreground/background samples.")
        else:
            for row in metrics.to_dict("records"):
                metric_name = str(row["metric"])
                st.metric(SEPARATION_LABELS.get(metric_name, metric_name), f"{float(row['value']):.3f}")
            _dataframe(st, metrics)


def _render_boundary_learning(st: object, px: object, run_path: Path, current_case_id: str) -> None:
    _render_page_guide(st, "boundary_learning")

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
    if image.ndim <= 2 or _is_rgb_image(image):
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
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32, copy=False)


def _rgb_base(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if _is_rgb_image(image):
        rgb = np.asarray(image[..., :3], dtype=np.float32)
        return _normalize(rgb)
    return np.repeat(_normalize(image)[..., None], 3, axis=-1)


def _is_rgb_image(image: np.ndarray) -> bool:
    image = np.asarray(image)
    return image.ndim == 3 and image.shape[-1] in {3, 4}


def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.38,
) -> np.ndarray:
    base = _rgb_base(image)
    mask_b = np.asarray(mask) > 0
    overlay_color = np.asarray(color, dtype=np.float32) / 255.0
    base[mask_b] = (1.0 - alpha) * base[mask_b] + alpha * overlay_color
    return base


def _error_overlay(image: np.ndarray, err: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    base = _rgb_base(image)
    err_i = np.asarray(err, dtype=np.int64)
    mask = err_i > 0
    colors = ERROR_COLORS[np.clip(err_i, 0, len(ERROR_COLORS) - 1)]
    base[mask] = (1.0 - alpha) * base[mask] + alpha * colors[mask]
    return base


def _render_page_guide(st: object, page: str) -> None:
    guide = PAGE_GUIDES.get(page)
    if not guide:
        return
    with st.expander("How to read this page / 如何阅读本页", expanded=False):
        st.markdown(guide.strip())


def _ordered_regions(regions: object) -> list[str]:
    region_set = {str(region) for region in regions}
    ordered = [region for region in REGION_COLORS if region in region_set]
    ordered.extend(sorted(region_set - set(ordered)))
    return ordered


def _default_regions_for_preset(all_regions: list[str], preset: str) -> list[str]:
    preferred = REGION_PRESETS.get(preset, tuple(all_regions))
    selected = [region for region in preferred if region in all_regions]
    return selected or list(all_regions)


def _nearest_epoch(available_epochs: list[int], requested_epoch: int) -> int:
    if not available_epochs:
        raise ValueError("available_epochs must not be empty")
    return min(
        available_epochs,
        key=lambda epoch: (abs(int(epoch) - int(requested_epoch)), int(epoch)),
    )


def _centroid_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["epoch", "region", "pc1", "pc2", "pc3", "samples"])
    return (
        df.groupby(["epoch", "region"], as_index=False)
        .agg(
            pc1=("pc1", "mean"),
            pc2=("pc2", "mean"),
            pc3=("pc3", "mean"),
            samples=("pc1", "size"),
        )
        .sort_values(["region", "epoch"])
    )


def _add_centroid_traces(
    fig: object,
    go: object,
    centroid_df: pd.DataFrame,
    regions: list[str],
    show_legend: bool,
) -> None:
    for region in regions:
        region_df = centroid_df[centroid_df["region"] == region].sort_values("epoch")
        if region_df.empty:
            continue
        color = REGION_COLORS.get(region, "#666666")
        fig.add_trace(
            go.Scatter3d(
                x=region_df["pc1"],
                y=region_df["pc2"],
                z=region_df["pc3"],
                mode="lines+markers",
                name=f"{region} centroid",
                showlegend=show_legend,
                line={"color": color, "width": 5},
                marker={"color": color, "size": 7, "symbol": "diamond"},
                hovertemplate=(
                    "region=%{customdata[0]}<br>"
                    "epoch=%{customdata[1]}<br>"
                    "samples=%{customdata[2]}<br>"
                    "PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>"
                ),
                customdata=region_df[["region", "epoch", "samples"]].to_numpy(),
            )
        )


def _add_region_surface_traces(
    fig: object,
    go: object,
    df: pd.DataFrame,
    regions: list[str],
    surface_mode: str,
) -> None:
    for region in regions:
        region_df = df[df["region"] == region]
        points = region_df[["pc1", "pc2", "pc3"]].dropna().to_numpy(dtype=np.float64)
        if points.shape[0] < 4 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
            continue

        color = REGION_COLORS.get(region, "#666666")
        trace_kwargs = {
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "name": f"{region} {surface_mode.lower()}",
            "color": color,
            "opacity": 0.14 if surface_mode == "Convex hull" else 0.10,
            "showlegend": True,
            "hoverinfo": "skip",
        }
        if surface_mode == "Convex hull":
            simplices = _convex_hull_simplices(points)
            if simplices.size == 0:
                continue
            trace_kwargs.update(
                {
                    "i": simplices[:, 0],
                    "j": simplices[:, 1],
                    "k": simplices[:, 2],
                    "flatshading": True,
                }
            )
        else:
            trace_kwargs["alphahull"] = 4
        fig.add_trace(go.Mesh3d(**trace_kwargs))


def _convex_hull_simplices(points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import ConvexHull, QhullError
    except Exception:
        return np.zeros((0, 3), dtype=np.int32)

    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(hull.simplices, dtype=np.int32)


def _feature_separation_metrics(df: pd.DataFrame, epoch: int) -> pd.DataFrame:
    current = df[df["epoch"] == epoch]
    if current.empty:
        return pd.DataFrame(columns=["metric", "value"])

    coord_columns = ["pc1", "pc2", "pc3"]
    centroids = current.groupby("region")[coord_columns].mean()
    rows: list[dict[str, float | str]] = []

    def centroid_distance(region_a: str, region_b: str) -> float | None:
        if region_a not in centroids.index or region_b not in centroids.index:
            return None
        a = centroids.loc[region_a, coord_columns].to_numpy(dtype=np.float32)
        b = centroids.loc[region_b, coord_columns].to_numpy(dtype=np.float32)
        return float(np.linalg.norm(a - b))

    boundary_within = None
    if "boundary" in centroids.index:
        boundary_points = current[current["region"] == "boundary"][coord_columns].to_numpy(
            dtype=np.float32
        )
        boundary_centroid = centroids.loc["boundary", coord_columns].to_numpy(dtype=np.float32)
        if boundary_points.size > 0:
            distances = np.linalg.norm(boundary_points - boundary_centroid, axis=1)
            boundary_within = float(distances.mean())
            rows.append({"metric": "boundary_within", "value": boundary_within})

    for metric, region_a, region_b in [
        ("boundary_to_foreground", "boundary", "foreground"),
        ("boundary_to_hard_background", "boundary", "hard_background"),
        ("fp_to_foreground", "false_positive", "foreground"),
        ("fn_to_foreground", "false_negative", "foreground"),
    ]:
        value = centroid_distance(region_a, region_b)
        if value is not None:
            rows.append({"metric": metric, "value": value})

    boundary_to_hard = centroid_distance("boundary", "hard_background")
    if boundary_to_hard is not None and boundary_within is not None:
        rows.append({"metric": "boundary_hard_margin", "value": boundary_to_hard - boundary_within})

    return pd.DataFrame(rows, columns=["metric", "value"])


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


def _plotly_chart(
    st: object,
    fig: object,
    key: str | None = None,
    selection_mode: str | None = None,
) -> object:
    extra_kwargs = {"key": key} if key else {}
    if selection_mode:
        extra_kwargs.update({"on_select": "rerun", "selection_mode": selection_mode})
    try:
        return st.plotly_chart(fig, width="stretch", **extra_kwargs)
    except TypeError:
        try:
            return st.plotly_chart(fig, use_container_width=True, **extra_kwargs)
        except TypeError:
            fallback_kwargs = {"key": key} if key else {}
            try:
                return st.plotly_chart(fig, width="stretch", **fallback_kwargs)
            except TypeError:
                return st.plotly_chart(fig, use_container_width=True, **fallback_kwargs)


def _dataframe(st: object, df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def _selected_point_id(selection_event: object) -> int | None:
    if selection_event is None:
        return None
    selection = _get_event_value(selection_event, "selection")
    points = _get_event_value(selection, "points") if selection is not None else None
    if not points:
        return None
    point = points[0]
    customdata = _get_event_value(point, "customdata")
    if customdata is None:
        customdata = _get_event_value(point, "custom_data")
    if customdata is None:
        point_index = _get_event_value(point, "point_index")
        return int(point_index) if point_index is not None else None
    if isinstance(customdata, np.ndarray):
        customdata = customdata.tolist()
    if isinstance(customdata, (list, tuple)) and customdata:
        return int(customdata[0])
    return int(customdata)


def _get_event_value(source: object, key: str) -> object | None:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _render_feature_point_detail(st: object, run_path: Path, row: pd.Series) -> None:
    case_id = str(row["case_id"])
    feature_coord = tuple(row["feature_coord_values"])
    spatial_shape = tuple(row["spatial_shape_values"])
    image_path = run_path / "cases" / sanitize_id(case_id) / "image.npy"
    image_coord: tuple[int, ...] | None = None
    image_preview = None

    if image_path.exists():
        image = load_array(image_path)
        image_coord = _feature_coord_to_image_coord(feature_coord, spatial_shape, image.shape)
        if image_coord is not None:
            image_preview = _point_overlay(image, image_coord)

    detail = pd.DataFrame(
        [
            {"field": "layer", "value": row["layer"]},
            {"field": "case_id", "value": case_id},
            {"field": "epoch", "value": int(row["epoch"])},
            {"field": "region", "value": row["region"]},
            {"field": "feature_coord", "value": row["feature_coord"]},
            {"field": "feature_shape", "value": row["spatial_shape"]},
            {
                "field": "image_coord",
                "value": _format_coord(image_coord) if image_coord is not None else "unavailable",
            },
        ]
    )
    _dataframe(st, detail)
    if image_preview is not None:
        st.image(image_preview, caption="Selected feature location", clamp=True)


def _feature_coord_to_image_coord(
    feature_coord: tuple[int, ...],
    spatial_shape: tuple[int, ...],
    image_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    image_spatial_shape = _image_spatial_shape(image_shape)
    dims = min(len(feature_coord), len(spatial_shape), len(image_spatial_shape))
    if dims == 0:
        return None
    coord = np.asarray(feature_coord[:dims], dtype=np.float64)
    source_shape = np.maximum(np.asarray(spatial_shape[:dims], dtype=np.float64), 1.0)
    target_shape = np.asarray(image_spatial_shape[:dims], dtype=np.float64)
    image_coord = np.floor((coord + 0.5) * target_shape / source_shape).astype(np.int64)
    image_coord = np.clip(image_coord, 0, target_shape.astype(np.int64) - 1)
    return tuple(int(value) for value in image_coord)


def _image_spatial_shape(image_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(image_shape) == 3 and image_shape[-1] in {3, 4}:
        return tuple(int(value) for value in image_shape[:2])
    return tuple(int(value) for value in image_shape)


def _point_overlay(image: np.ndarray, image_coord: tuple[int, ...]) -> np.ndarray | None:
    base = _rgb_base(image)
    if base.ndim != 3 or len(image_coord) < 2:
        return None
    y, x = int(image_coord[0]), int(image_coord[1])
    height, width = base.shape[:2]
    if not (0 <= y < height and 0 <= x < width):
        return None
    radius = max(3, min(height, width) // 40)
    color = np.asarray([1.0, 0.1, 0.85], dtype=np.float32)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    base[y0:y1, x] = color
    base[y, x0:x1] = color
    return base


def _format_coord(coord: tuple[int, ...] | None) -> str:
    if coord is None:
        return "unavailable"
    return ",".join(str(int(value)) for value in coord)


def _feature_space_html_name(layer: str, epoch: int, region_preset: str) -> str:
    safe_layer = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in layer)
    safe_preset = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in region_preset)
    return f"segevo_feature_space_{safe_layer}_epoch{int(epoch):04d}_{safe_preset}.html"


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
    feature_coord_values = [tuple(int(value) for value in row) for row in projection.feature_coords]
    spatial_shapes = getattr(projection, "feature_spatial_shapes", None)
    if spatial_shapes is None:
        spatial_shapes = np.zeros_like(np.asarray(projection.feature_coords))
    spatial_shape_values = [tuple(int(value) for value in row) for row in spatial_shapes]
    spatial_shape_labels = [_format_coord(values) for values in spatial_shape_values]
    return pd.DataFrame(
        {
            "point_id": np.arange(len(projection.x), dtype=np.int64),
            "layer": projection.layer,
            "pc1": projection.x,
            "pc2": projection.y,
            "pc3": projection.z,
            "region": region_labels,
            "case_id": projection.case_ids,
            "epoch": projection.epochs,
            "epoch_label": projection.epochs.astype(str),
            "coord": coords,
            "feature_coord": coords,
            "feature_coord_values": feature_coord_values,
            "spatial_shape": spatial_shape_labels,
            "spatial_shape_values": spatial_shape_values,
        }
    )
