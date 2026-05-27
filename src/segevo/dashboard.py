"""Streamlit dashboard for SegEvo artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from segevo.artifacts import list_cases, list_epochs, load_array, read_manifest, sanitize_id
from segevo.boundary_learning import boundary_learning_records
from segevo.feature_space import (
    PROJECTION_METHODS,
    ProjectionUnavailableError,
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

CASE_METRIC_ORDER = ("dice", "surface_dice", "hd95", "volume_error")
BOUNDARY_METRIC_ORDER = (
    "boundary_dice",
    "surface_dice",
    "hd95",
    "boundary_to_hard_background",
    "boundary_hard_margin",
)
LOWER_IS_BETTER = {"hd95"}
CLOSER_TO_ZERO_IS_BETTER = {"volume_error"}

PAGE_GUIDES = {
    "case_timeline": """
**English**

**What this page is for**

Use this page when you want to answer a simple question: for one fixed probe case,
what did the model predict at this epoch, where is it wrong, and is the case getting
better as training continues?

**How to use it**

1. Choose `Case` in the sidebar. A case is usually one validation image, slice, tile,
   or 3D volume that you log repeatedly during training.
2. Move the sidebar `Epoch` slider. All panels on this page switch to the same epoch,
   so you can replay the model's behavior over time.
3. Compare the three image panels first, then read the metric curve below.
4. Use `Training Readout` for a quick current-epoch summary before digging into the
   full timeline.

**How to read the panels**

- `Image + GT`: green is the ground-truth annotation. This is the target the model
  should learn.
- `Image + Prediction`: blue is the model's prediction at the selected epoch.
  If blue gradually overlaps green as you move epochs, the model is learning this case.
- `Error Map`: green means correct foreground, orange means false positive
  (the model predicted too much), and red means false negative (the model missed
  real foreground).
- `Training Readout`: current metric cards show the selected epoch and how much each
  metric changed from the first logged epoch. The error-balance table tells you whether
  mistakes are dominated by FP, FN, or both.
- `Metrics Timeline`: Dice and surface Dice should go up. HD95 should go down.
  Volume error should move toward zero. When the curve improves but the error map
  still has red/orange regions, the model is improving globally but still has local
  failure modes.
- `Feature Summaries` and `Feature Sample Counts`: these confirm that SegEvo actually
  captured intermediate layer features for this epoch. If this section is missing,
  the training script may not have attached feature layers or logged probe features.

**What you can learn about training**

- Whether training is actually improving this case, not just improving the average
  validation score.
- Whether errors are mostly false positives, false negatives, or boundary shifts.
- Whether improvement is smooth or unstable across epochs. A curve that jumps up and
  down often means this probe case is sensitive to the current training setup.
- Whether a metric improvement matches the visual result. If Dice rises but the error
  map still shows the same red/orange area, the model may be learning easy regions while
  ignoring the hard part.

**中文**

**这个页面用来回答什么**

这个页面适合先看一个最直观的问题：对同一个固定 probe 病例，模型在当前 epoch
预测成什么样、错在哪里、随着训练推进有没有变好。

**怎么操作**

1. 先在左侧 `Case` 选择一个病例。这里的 case 通常是你训练时反复记录的一张验证图、
   一个切片、一个 WSI tile，或者一个 3D volume。
2. 再拖动左侧 `Epoch`。这个页面所有内容都会切到同一个 epoch，所以可以像回放一样看
   模型从早期到后期的变化。
3. 阅读顺序建议是：先看上面的三张图，再看下面的指标曲线。
4. `Training Readout` 是当前 epoch 的快速摘要，建议先看它，再细看曲线。

**怎么看图和指标**

- `Image + GT`：绿色是真实标注，也就是模型应该学到的目标。
- `Image + Prediction`：蓝色是模型当前 epoch 的预测。如果拖动 epoch 时蓝色逐渐贴近绿色，
  说明这个病例在变好。
- `Error Map`：绿色表示前景预测对了，橙色表示假阳性 FP，也就是模型多分出来了；
  红色表示假阴性 FN，也就是模型漏掉了真实前景。
- `Training Readout`：指标卡展示当前 epoch 的数值，以及相对第一个记录 epoch 的变化；
  error-balance 表格可以快速判断错误主要来自 FP、FN，还是两者都有。
- `Metrics Timeline`：Dice 和 surface Dice 越高越好；HD95 越低越好；
  volume error 越接近 0 越好。如果曲线在变好，但 Error Map 仍然有红色或橙色区域，
  说明整体在进步，但局部还有稳定错误。
- `Feature Summaries` 和 `Feature Sample Counts`：用来确认当前 epoch 是否真的记录到了中间层
  feature。如果这里没有内容，通常说明训练脚本还没有 attach layer，或者没有把 probe feature
  写进 SegEvo run。

**从这里能读出哪些训练信息**

- 这个病例本身有没有真的变好，而不只是整体 validation score 变好了。
- 错误主要是假阳性、多分了，还是假阴性、漏分了，或者主要是边界偏移。
- 训练过程是否稳定。如果曲线反复上下跳，说明这个 probe case 对当前训练设置比较敏感。
- 指标变好是否真的对应视觉变好。如果 Dice 上升了，但 Error Map 里同一块红色或橙色一直存在，
  说明模型可能只学到了容易区域，困难区域还没解决。
""",
    "feature_space": """
**English**

**What this page is for**

This page does not show the mask directly. It shows how the model represents different
types of pixels or voxels inside a selected layer. SegEvo samples foreground, boundary,
hard background, false-positive, and false-negative locations, takes their layer features,
and projects those high-dimensional vectors into a 3D plot with PCA, UMAP, or t-SNE.

The projection axes are not image coordinates. Do not ask whether a projection axis
means left/right or bright/dark. Instead, ask: are useful regions forming separate
groups, are error points moving closer to correct regions, and are centroid trajectories
stabilizing across epochs?

**How to use it**

1. `Layer`: choose which hooked network layer to inspect. Early layers usually respond
   to texture, stain, edge, and intensity. Deeper layers usually carry more class or
   shape information.
2. `Projection`: keep `PCA 3D` as the default for stable epoch-to-epoch comparison.
   `UMAP 3D` and `t-SNE 3D` are better for exploring local cluster structure.
3. `Epoch`: this is synced with the sidebar epoch. If the sidebar says epoch 6, this
   page also focuses on epoch 6.
4. `Region preset`: start here instead of manually clicking many labels.
   `All regions` shows everything, `Core regions` shows foreground/boundary/hard
   background, `Errors only (FP/FN)` isolates mistakes, and `Boundary focus` is useful
   when debugging edge errors.
5. `Display`: use `Points + centroid trajectories` for the normal view. `Points only`
   is cleaner when there are many samples. `Centroids only` is best when you want to
   see how each region moves through training.
6. `Epoch playback`: turn this on when you want an animation from early epochs to the
   current sidebar epoch.
7. `Surface`: `Convex hull` wraps a region with a transparent shell; `Density cloud`
   gives a softer volume-like impression. These are most useful for foreground,
   boundary, and hard background.
8. Click any point to fill `Point Detail`. It tells you the point's case, epoch, region,
   feature coordinate, and mapped image location.
9. `Export 3D HTML` downloads the interactive plot for reports, slides, papers, or
   GitHub issues.

Projection rule of thumb: PCA is more stable, while UMAP/t-SNE are better for exploring
local cluster structure. Treat UMAP/t-SNE as exploratory views, not exact global maps.

**How to interpret common patterns**

- Good sign: foreground, boundary, and hard background become more separated as epochs
  increase.
- Good sign: FP/FN points shrink in number or move closer to the correct foreground or
  boundary cluster.
- Warning sign: boundary and hard background remain mixed even when Dice is high. The
  model may look good globally but still be unstable around edges.
- Warning sign: centroid trajectories jump around late in training. The representation
  may still be unstable, or the learning rate may be too high.
- `Separation` gives a quick numeric summary of these distances. Larger
  boundary-to-hard-background distance is usually better; smaller FP/FN-to-foreground
  distance can mean errors are becoming less severe.

**What you can learn about training**

- Whether the network representation is becoming more organized. Good training often
  turns mixed feature clouds into clearer region clusters.
- Whether boundary learning is happening internally before it is obvious in Dice.
  Boundary points moving away from hard background can be an early sign of improvement.
- Whether the model is confusing foreground with nearby background. If foreground,
  boundary, and hard background stay tangled, the model may need better data, loss
  weighting, augmentation, or more boundary-focused supervision.
- Whether FP/FN errors are random or systematic. A compact FP/FN cluster means the model
  is making a repeatable kind of mistake; scattered FP/FN points often mean errors come
  from multiple causes.
- Which layer is most diagnostic. If deeper layers separate regions clearly but shallow
  layers do not, the model has learned a semantic decision. If no layer separates them,
  the training signal may be too weak for this task.

**中文**

**这个页面用来回答什么**

这个页面不是直接看 mask，而是看“模型内部怎么看这些像素/体素”。SegEvo 会从
foreground、boundary、hard background、false positive、false negative 这些区域采样，
取出它们在某个网络层里的 feature vector，然后用 PCA、UMAP 或 t-SNE 压到三维空间里画出来。

这里的投影坐标不是原图坐标，不代表图像里的上下左右，也不直接代表亮度或颜色。更应该看的问题是：
不同区域的点群有没有分开，错误点有没有往正确点群靠近，中心轨迹到后期是否越来越稳定。

**怎么操作**

1. `Layer`：选择要看的网络层。浅层通常更像是在看纹理、染色、边缘、强度；深层通常更接近
   类别、形状和语义。
2. `Projection`：默认使用 `PCA 3D`，它更适合稳定地比较不同 epoch。`UMAP 3D` 和
   `t-SNE 3D` 更适合探索局部聚类结构。
3. `Epoch`：这里跟左侧全局 epoch 同步。左侧是 epoch 6，这里看的也是 epoch 6。
4. `Region preset`：建议先用这个，不用一上来手动点很多标签。`All regions` 看全部；
   `Core regions` 看 foreground / boundary / hard background；`Errors only (FP/FN)` 只看错误点；
   `Boundary focus` 适合排查边界问题。
5. `Display`：常规建议用 `Points + centroid trajectories`。点太多时用 `Points only` 更清爽；
   想看训练过程中的区域移动趋势时，用 `Centroids only`。
6. `Epoch playback`：打开后可以播放从早期到当前左侧 epoch 的点云变化。
7. `Surface`：`Convex hull` 会给点群包一层透明外壳；`Density cloud` 更像一个柔和的密度云。
   它们主要用于看 foreground、boundary、hard background 的覆盖范围。
8. 点击 3D 图里的任意点，右侧 `Point Detail` 会显示它来自哪个 case、epoch、region、
   feature 坐标，以及映射回原图的大致位置。
9. `Export 3D HTML` 可以把当前交互式 3D 图导出，后续放到汇报、论文补充材料或 GitHub issue
   里都比较方便。

投影方式的经验判断：PCA 更稳定，UMAP/t-SNE 更适合探索局部聚类结构。UMAP/t-SNE 更适合辅助发现
局部簇，不建议把它们当成严格的全局距离地图。

**常见现象怎么判断**

- 好现象：随着 epoch 增加，foreground、boundary、hard background 逐渐分开。
- 好现象：FP/FN 点变少，或者逐渐靠近正确的 foreground / boundary 点群，说明错误在变轻。
- 需要注意：Dice 已经不错，但 boundary 和 hard background 仍然混在一起。这说明整体重叠还行，
  但边界附近可能不稳定。
- 需要注意：训练后期中心轨迹还在大幅跳动，说明 feature 表征可能还没稳定，也可能学习率偏大。
- `Separation` 是这些距离的数字摘要。boundary 到 hard background 越远通常越好；
  FP/FN 到 foreground 越近，有时说明错误点正在变得更像正确前景，错误严重程度在下降。

**从这里能读出哪些训练信息**

- 网络内部表征有没有变得更有组织。比较好的训练通常会让混在一起的点云逐渐形成更清楚的区域簇。
- 边界学习是不是已经在内部发生了。有时 Dice 还没明显变化，但 boundary 点已经开始远离
  hard background，这可能是后续边界变好的早期信号。
- 模型是否把前景和附近背景混淆。如果 foreground、boundary、hard background 一直缠在一起，
  可能需要更合适的数据、loss 权重、augmentation，或者更强调边界的监督。
- FP/FN 是随机错误还是系统性错误。如果 FP/FN 形成紧凑点群，说明模型在稳定地犯某一类错误；
  如果 FP/FN 很分散，说明错误来源可能比较复杂。
- 哪一层最有诊断价值。如果深层能把区域分开、浅层分不开，说明模型形成了语义判断；
  如果所有层都分不开，可能是这个任务的训练信号还不够强。
""",
    "boundary_learning": """
**English**

**What this page is for**

Dice can look acceptable even when the predicted edge is rough, shifted, or unstable.
This page focuses on that edge behavior. It asks: is the model learning the object
boundary, are surface errors shrinking, and are boundary features separating from
foreground and confusing background?

**How to use it**

1. `Boundary cases`: use `Current case` when debugging one probe case. Use `All cases`
   when you want an average trend across the logged run.
2. `Boundary layer`: choose which feature layer to use for boundary-separation analysis.
   A deeper layer often gives a cleaner semantic boundary; a shallower layer may reveal
   texture-driven confusion.
3. `Boundary width`: controls how thick the boundary band is. A small value is stricter
   and focuses on the exact edge. A larger value is more forgiving and useful when masks
   are noisy or annotation boundaries are not pixel-perfect.
4. `Surface tolerance`: controls how much distance error is accepted by surface Dice.
   Smaller tolerance is stricter. Larger tolerance asks whether the prediction is at
   least close to the target surface.
5. In `Boundary metrics`, select the curves you care about. Start with
   `boundary_dice`, `surface_dice`, and `hd95`.
6. Read `Boundary Readout` first. It compares the latest logged epoch with the first
   logged epoch and translates the curves into a short training diagnosis.

**How to interpret the curves**

- `boundary_dice`: higher is better. It means the predicted boundary overlaps the
  ground-truth boundary band more closely.
- `surface_dice`: higher is better. It means the predicted surface is within the chosen
  distance tolerance.
- `hd95`: lower is better. It summarizes large boundary errors while being less sensitive
  than the maximum Hausdorff distance.
- `Boundary Feature Separation`: this table and trend compare boundary features against
  foreground and hard background. If boundary-to-hard-background distance rises, the
  model is usually learning to tell the true edge apart from nearby confusing tissue.
- `Boundary Readout`: compact metric cards show the latest boundary state. The diagnosis
  bullets summarize whether overlap, surface distance, and feature separation improved.
- A useful pattern is: boundary Dice goes up, HD95 goes down, and boundary-to-hard-
  background separation goes up. That usually means both the output mask and the internal
  boundary representation are improving.

**What you can learn about training**

- Whether the model is only learning coarse object overlap or also learning usable
  boundaries.
- Whether a high Dice score is hiding boundary problems. High Dice with poor HD95 or
  weak boundary separation means the model may be good in the center but unreliable at
  the edge.
- Whether boundary errors are getting smaller over time. Surface Dice rising and HD95
  falling is usually more meaningful for clinical or pathology edge quality than Dice
  alone.
- Whether boundary representations are stable enough for deployment-like data. If
  boundary feature separation does not improve, the model may fail on hard negatives
  near the target tissue.

**中文**

**这个页面用来回答什么**

有时候 Dice 看起来还可以，但边界其实很毛糙、偏移，或者某些区域反复分错。这个页面专门看
边界学习：模型有没有学到真实边缘，surface 误差有没有变小，边界特征能不能和前景内部、
易混背景区分开。

**怎么操作**

1. `Boundary cases`：调试单个 probe 病例时选 `Current case`；想看整体趋势时选 `All cases`。
2. `Boundary layer`：选择用于分析边界特征的网络层。深层通常更偏语义边界；浅层可能更容易暴露
   纹理、染色、强度导致的混淆。
3. `Boundary width`：控制边界带有多宽。数值小更严格，主要看精确边缘；数值大更宽松，
   适合标注边缘本身比较粗糙或不完全像素级准确的数据。
4. `Surface tolerance`：控制 surface Dice 接受多大的边界距离误差。数值小更严格；
   数值大则是在问“预测边界是否至少离真实边界不远”。
5. `Boundary metrics` 里可以选择要看的曲线。建议先看 `boundary_dice`、`surface_dice` 和 `hd95`。
6. 先看 `Boundary Readout`。它会把最新 epoch 和第一个记录 epoch 做对比，并把曲线翻译成
   简短的训练诊断。

**怎么看曲线和表格**

- `boundary_dice`：越高越好，表示预测边界和真实边界带重叠得更好。
- `surface_dice`：越高越好，表示预测 surface 在设定容忍距离内的比例更高。
- `hd95`：越低越好，它反映较大的边界误差，但比最大 Hausdorff distance 更不容易被极端离群点影响。
- `Boundary Feature Separation`：看 boundary feature 和 foreground / hard background feature
  是否分开。boundary 到 hard background 距离上升，通常说明模型更能区分真实边界和附近易混背景。
- `Boundary Readout`：用紧凑指标卡展示最新边界状态，并用诊断语句总结边界重叠、surface 距离、
  feature separation 是否改善。
- 一个比较理想的训练趋势是：boundary Dice 上升，HD95 下降，同时 boundary-to-hard-background
  separation 上升。这说明输出 mask 的边界和模型内部的边界表征都在变好。

**从这里能读出哪些训练信息**

- 模型只是学到了粗略重叠，还是也学到了可用的边界。
- 高 Dice 是否掩盖了边界问题。如果 Dice 高，但 HD95 差，或者 boundary separation 很弱，
  说明模型可能在目标中心区域还不错，但边缘不可靠。
- 边界误差是否真的随训练变小。对于临床图像或病理分割，surface Dice 上升、HD95 下降通常
  比只看 Dice 更能反映边界质量。
- 边界表征是否足够稳定。如果 boundary feature separation 一直没有改善，模型在真实数据里
  遇到目标附近的 hard negative 时可能仍然容易出错。
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

    case_metrics = _case_metric_frame(metrics, case_id)
    _render_case_training_readout(st, case_metrics, epoch, err)

    if not metrics.empty:
        st.subheader("Metrics Timeline")
        metric_names = [
            name
            for name in CASE_METRIC_ORDER
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
        projection_method = st.selectbox(
            "Projection",
            list(PROJECTION_METHODS),
            index=0,
            help=(
                "PCA is the most stable for epoch-to-epoch comparison. "
                "UMAP/t-SNE are better for exploring local cluster structure."
            ),
        )
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

    try:
        projection = project_feature_space(space, method=projection_method)
    except ProjectionUnavailableError as exc:
        st.warning(str(exc))
        return
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

    title = _projection_title(projection)
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
            "xaxis_title": projection.axis_names[0],
            "yaxis_title": projection.axis_names[1],
            "zaxis_title": projection.axis_names[2],
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
            file_name=_feature_space_html_name(
                layer,
                focus_epoch,
                region_preset,
                projection.method,
            ),
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
        _render_boundary_training_readout(st, plot_df)

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


def _case_metric_frame(metrics: pd.DataFrame, case_id: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    if "case_id" not in metrics.columns:
        return metrics.copy()
    case_metrics = metrics[metrics["case_id"].astype(str) == str(case_id)].copy()
    return case_metrics if not case_metrics.empty else metrics.copy()


def _render_case_training_readout(
    st: object,
    case_metrics: pd.DataFrame,
    epoch: int,
    err: np.ndarray,
) -> None:
    st.subheader("Training Readout")
    metric_rows = _metric_readout_rows(case_metrics, CASE_METRIC_ORDER, focus_epoch=epoch)
    if metric_rows:
        _render_metric_cards(st, metric_rows)

    error_rows = _error_balance_rows(err)
    if error_rows:
        st.caption(_error_balance_sentence(error_rows))
        _dataframe(st, pd.DataFrame(error_rows))


def _render_boundary_training_readout(st: object, plot_df: pd.DataFrame) -> None:
    st.subheader("Boundary Readout")
    metric_rows = _metric_readout_rows(plot_df, BOUNDARY_METRIC_ORDER)
    if not metric_rows:
        st.info("No numeric boundary metrics available for this selection.")
        return

    _render_metric_cards(st, metric_rows)
    diagnosis = _boundary_diagnosis(metric_rows)
    if diagnosis:
        st.markdown("\n".join(f"- {item}" for item in diagnosis))


def _render_metric_cards(st: object, rows: list[dict[str, object]]) -> None:
    cols = st.columns(min(len(rows), 4))
    for index, row in enumerate(rows):
        col = cols[index % len(cols)]
        with col:
            try:
                st.metric(
                    str(row["metric"]),
                    _format_metric_value(float(row["value"])),
                    delta=str(row["delta"]),
                    delta_color=str(row["delta_color"]),
                )
            except TypeError:
                st.metric(
                    str(row["metric"]),
                    _format_metric_value(float(row["value"])),
                    delta=str(row["delta"]),
                )
            st.caption(str(row["status"]))


def _metric_readout_rows(
    df: pd.DataFrame,
    metric_names: tuple[str, ...],
    focus_epoch: int | None = None,
) -> list[dict[str, object]]:
    if df.empty or "epoch" not in df.columns:
        return []

    available = [metric for metric in metric_names if metric in df.columns]
    if not available:
        return []

    numeric = df[["epoch", *available]].copy()
    numeric["epoch"] = pd.to_numeric(numeric["epoch"], errors="coerce")
    for metric in available:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    numeric = numeric.dropna(subset=["epoch"])
    if numeric.empty:
        return []

    by_epoch = (
        numeric.groupby("epoch", as_index=False)[available]
        .mean(numeric_only=True)
        .sort_values("epoch")
    )
    if by_epoch.empty:
        return []

    epochs = by_epoch["epoch"].to_numpy(dtype=np.float64)
    selected_index = (
        int(np.argmin(np.abs(epochs - float(focus_epoch))))
        if focus_epoch is not None
        else len(by_epoch) - 1
    )
    current = by_epoch.iloc[selected_index]
    first = by_epoch.iloc[0]

    rows: list[dict[str, object]] = []
    for metric in available:
        value = float(current[metric])
        first_value = float(first[metric])
        if not np.isfinite(value):
            continue
        status = _metric_status(metric, value, first_value)
        delta = value - first_value if np.isfinite(first_value) else np.nan
        rows.append(
            {
                "metric": metric,
                "epoch": int(current["epoch"]),
                "value": value,
                "first_value": first_value,
                "delta_value": delta,
                "delta": _format_metric_delta(delta),
                "status": status,
                "delta_color": _metric_delta_color(metric),
            }
        )
    return rows


def _metric_status(metric: str, value: float, baseline: float, tol: float = 1e-6) -> str:
    if not np.isfinite(baseline):
        return "baseline unavailable"
    if metric in LOWER_IS_BETTER:
        if value < baseline - tol:
            return "improved vs first epoch"
        if value > baseline + tol:
            return "worse vs first epoch"
        return "stable vs first epoch"
    if metric in CLOSER_TO_ZERO_IS_BETTER:
        if abs(value) < abs(baseline) - tol:
            return "closer to target than first epoch"
        if abs(value) > abs(baseline) + tol:
            return "farther from target than first epoch"
        return "stable vs first epoch"
    if value > baseline + tol:
        return "improved vs first epoch"
    if value < baseline - tol:
        return "worse vs first epoch"
    return "stable vs first epoch"


def _metric_delta_color(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "inverse"
    if metric in CLOSER_TO_ZERO_IS_BETTER:
        return "off"
    return "normal"


def _format_metric_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_metric_delta(delta: float) -> str:
    if not np.isfinite(delta):
        return "vs first n/a"
    return f"vs first {delta:+.3f}"


def _error_balance_rows(err: np.ndarray) -> list[dict[str, object]]:
    err_i = np.asarray(err, dtype=np.int64)
    tp = int(np.count_nonzero(err_i == 1))
    fp = int(np.count_nonzero(err_i == 2))
    fn = int(np.count_nonzero(err_i == 3))
    pred_foreground = tp + fp
    gt_foreground = tp + fn
    precision = tp / pred_foreground if pred_foreground else np.nan
    recall = tp / gt_foreground if gt_foreground else np.nan
    total_error = fp + fn
    total_labeled = tp + total_error

    rows = [
        ("true_positive", tp, "correct foreground"),
        ("false_positive", fp, "over-segmentation"),
        ("false_negative", fn, "missed foreground"),
        ("precision_proxy", precision, "TP / (TP + FP)"),
        ("recall_proxy", recall, "TP / (TP + FN)"),
    ]
    output: list[dict[str, object]] = []
    for name, value, meaning in rows:
        if isinstance(value, float):
            display_value = value
            fraction = value
        else:
            display_value = int(value)
            fraction = value / total_labeled if total_labeled else np.nan
        output.append(
            {
                "signal": name,
                "value": display_value,
                "fraction": fraction,
                "meaning": meaning,
            }
        )
    return output


def _error_balance_sentence(rows: list[dict[str, object]]) -> str:
    values = {str(row["signal"]): row["value"] for row in rows}
    fp = int(values.get("false_positive", 0))
    fn = int(values.get("false_negative", 0))
    if fp == 0 and fn == 0:
        return "No FP/FN pixels at this epoch for the logged mask."
    if fp > fn * 1.5:
        return "Current errors are FP-dominant: the model is mostly over-segmenting."
    if fn > fp * 1.5:
        return "Current errors are FN-dominant: the model is mostly missing foreground."
    return "Current errors are balanced between FP and FN."


def _boundary_diagnosis(rows: list[dict[str, object]]) -> list[str]:
    by_metric = {str(row["metric"]): str(row["status"]) for row in rows}
    messages: list[str] = []
    if by_metric.get("boundary_dice", "").startswith("improved"):
        messages.append("Boundary overlap improved compared with the first logged epoch.")
    elif by_metric.get("boundary_dice", "").startswith("worse"):
        messages.append("Boundary overlap is worse than the first logged epoch.")

    if by_metric.get("hd95", "").startswith("improved"):
        messages.append("Large boundary errors are shrinking because HD95 went down.")
    elif by_metric.get("hd95", "").startswith("worse"):
        messages.append("Large boundary errors increased because HD95 went up.")

    if by_metric.get("boundary_to_hard_background", "").startswith("improved"):
        messages.append("Boundary features are separating from hard background.")
    elif by_metric.get("boundary_to_hard_background", "").startswith("worse"):
        messages.append("Boundary features are less separated from hard background.")

    return messages


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


def _projection_title(projection: object) -> str:
    method = str(getattr(projection, "method", "PCA 3D"))
    layer = str(getattr(projection, "layer", "layer"))
    ratios = tuple(float(value) for value in getattr(projection, "explained_variance_ratio", ()))
    if method == "PCA 3D" and len(ratios) >= 3:
        return (
            f"{layer} 3D PCA "
            f"(PC1 {ratios[0]:.1%}, PC2 {ratios[1]:.1%}, PC3 {ratios[2]:.1%})"
        )
    return f"{layer} {method}"


def _feature_space_html_name(
    layer: str,
    epoch: int,
    region_preset: str,
    projection_method: str = "PCA 3D",
) -> str:
    safe_layer = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in layer)
    safe_preset = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in region_preset)
    safe_method = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in projection_method.lower().replace(" ", "_")
    )
    return (
        f"segevo_feature_space_{safe_layer}_{safe_method}_"
        f"epoch{int(epoch):04d}_{safe_preset}.html"
    )


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
