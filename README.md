# SegEvo

[![CI](https://github.com/DZW131/SegEvo/actions/workflows/ci.yml/badge.svg)](https://github.com/DZW131/SegEvo/actions/workflows/ci.yml)

**Language:** English | [简体中文](README.zh-CN.md)

中文定位：一个面向医学影像分割的训练动态可视化工具，用于追踪预测 mask、错误区域、中间特征、边界表征、注意力/不确定性在训练过程中的演化。

SegEvo is an open-source research tool for observing how medical image segmentation
models learn during training. It records fixed probe cases across epochs and replays
the evolution of predictions, error regions, metrics, and sampled intermediate
representations in a lightweight dashboard.

The project is intentionally a low-intrusion observation layer, not a new training
framework. A typical PyTorch segmentation workflow should only need a few lines to
write SegEvo artifacts.

## Why SegEvo

Segmentation quality is usually summarized by final Dice, HD95, or surface Dice. SegEvo
focuses on the process:

- When does the model move from coarse localization to boundary refinement?
- Which cases remain unstable or repeatedly forgotten during training?
- Are false positives and false negatives visible early in feature space?
- Do boundary and hard-background representations separate over time?

## Quick Start

```bash
git clone https://github.com/DZW131/SegEvo.git
cd SegEvo
pip install -e ".[dashboard]"

segevo-demo --out runs/demo
segevo-dashboard --run runs/demo --host 0.0.0.0 --port 7860
```

On a remote server, forward the port from your local machine:

```bash
ssh -L 7860:localhost:7860 user@server
```

Then open `http://localhost:7860`.

If you are installing on an older Linux research server, upgrade packaging tools
first so pip can use prebuilt wheels:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dashboard]"
```

SegEvo pins the dashboard `pyarrow` dependency below `21` because newer wheels can
fall back to source builds on older glibc systems, which may fail with Rust/C++
compiler errors.

The dashboard currently has three main tabs:

- `Case Timeline`: image, GT, prediction, FP/FN error map, metrics, and feature
  sample counts for the selected case and epoch.
- `Feature Space`: stable PCA projection of sampled layer features across selected
  cases and epochs, colored by foreground, boundary, hard background, FP, and FN.
- `Boundary Learning`: boundary Dice, surface Dice, HD95, and boundary feature
  separation trends for selected cases and layers.

## PyTorch U-Net Example

SegEvo includes a real PyTorch training example with a tiny U-Net and a synthetic
lesion-like dataset. It is intentionally small enough to run on CPU, but it follows
the same pattern as a normal segmentation project:

```bash
pip install -e ".[torch,dashboard]"
python examples/pytorch_unet_training.py --epochs 3 --run-dir runs/pytorch_unet
segevo-dashboard --run runs/pytorch_unet --host 0.0.0.0 --port 7860
```

The SegEvo-specific integration is only these steps:

```python
logger = SegEvoLogger(run_dir="runs/pytorch_unet", manifest={...})
logger.attach(model, layers=["enc2", "bottleneck", "dec1"])

# inside validation or probe-case logging
logger.log_case(
    epoch=epoch,
    case_id=case_id,
    image=image,
    gt=gt,
    pred=pred,
    uncertainty=uncertainty,
    metrics={"train_loss": train_loss, "probe_dice": probe_dice},
)
```

## Minimal Training Integration

For the first version, the most stable API is explicit logging. You can use it from
any training loop as long as you can provide image, ground truth, and prediction arrays.

```python
from segevo import SegEvoLogger

logger = SegEvoLogger(
    run_dir="runs/liver_unet",
    manifest={
        "project": "liver_unet",
        "task": "binary_liver_segmentation",
        "classes": ["background", "liver"],
        "spacing": [1.0, 1.0, 1.0],
    },
)

for epoch in range(num_epochs):
    train_one_epoch(...)
    metrics = validate(...)

    if epoch % 5 == 0:
        for case in probe_cases:
            image, gt = case["image"], case["mask"]
            pred = infer(model, image)

            logger.log_case(
                epoch=epoch,
                case_id=case["case_id"],
                image=image,
                gt=gt,
                pred=pred,
                metrics={"val_dice": metrics["dice"]},
            )
```

If you want to record layer summaries from a PyTorch model:

```python
logger.attach(model, layers=["encoder.3", "bottleneck", "decoder.2"])
```

The first MVP stores compact feature statistics by default. Dense feature-volume
storage is deliberately avoided because 3D medical data can become very large.

## Artifact Layout

SegEvo dashboards read a run directory and do not depend on the training script.

```text
runs/liver_unet/
  manifest.json
  metrics.csv
  cases/
    case_001/
      image.npy
      gt.npy
      epochs/
        0000/
          pred.npy
          error.npy
          features.npz
        0005/
          pred.npy
          error.npy
```

`error.npy` uses compact integer labels:

- `0`: true background
- `1`: true positive
- `2`: false positive
- `3`: false negative

`features.npz` stores compact layer data. For a logged layer such as `bottleneck`,
the current schema writes:

- `bottleneck__summary`: mean, standard deviation, minimum, and maximum activation.
- `bottleneck__samples`: sampled feature vectors with shape `[N, C]`.
- `bottleneck__sample_region_ids`: integer region labels for each sampled vector.
- `bottleneck__sample_coords`: feature-map coordinates for each sampled vector.
- `feature_region_names`: region-id names in order.

The five sampled regions are `foreground`, `boundary`, `hard_background`,
`false_positive`, and `false_negative`.

## Current MVP Scope

- Run format with `manifest.json`, `metrics.csv`, and per-case epoch artifacts.
- Explicit case logging for 2D and 3D binary segmentation masks.
- Dice, volume error, HD95, and surface Dice helpers.
- Synthetic demo run generator.
- Streamlit dashboard for case timeline, slice viewer, error overlay, and metric curves.
- Optional PyTorch forward hooks for compact layer activation summaries and sampled
  feature vectors.
- Feature sampling for foreground, boundary, hard background, FP, and FN pixels or
  voxels.
- Feature Space dashboard tab with stable PCA projection across selected cases and
  epochs.
- Boundary Learning dashboard tab with boundary metrics and boundary-vs-background
  feature separation.

## Roadmap

- UMAP feature-space replay across epochs and layers.
- Failure explorer for unstable, forgotten, and persistently wrong cases.
- More PyTorch examples for common 2D and 3D segmentation loops.
- Uncertainty and attention/Grad-CAM artifact conventions.

## Development

```bash
pip install -e ".[dashboard,dev]"
pytest
ruff check .
```

GitHub Actions runs linting, unit tests, and a tiny PyTorch U-Net smoke test on
every push and pull request.

If you are working on a server without browser access, generate a demo run and start
the dashboard on `0.0.0.0`, then use SSH port forwarding as shown above.
