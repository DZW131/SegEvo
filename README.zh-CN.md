# SegEvo

[![CI](https://github.com/DZW131/SegEvo/actions/workflows/ci.yml/badge.svg)](https://github.com/DZW131/SegEvo/actions/workflows/ci.yml)

**语言:** [English](README.md) | 简体中文

SegEvo 是一个面向医学影像分割的开源科研辅助工具，用于观察分割模型在训练过程中如何逐步学习。它会记录固定 probe 病例在不同 epoch 的预测结果、错误区域、指标变化、中间层特征采样，并在轻量 dashboard 中回放这些变化。

这个项目的定位不是新的训练框架，而是一个低侵入的观测层。对于普通 PyTorch 分割训练代码，理想情况下只需要加几行日志代码，就能生成 SegEvo 可视化所需的实验产物。

## 为什么做 SegEvo

医学图像分割通常只看最终 Dice、HD95、surface Dice 等结果指标。但科研中更有价值的问题往往是训练过程本身：

- 模型什么时候从粗定位进入边界精修？
- 哪些病例一直不稳定，或者在训练中反复被遗忘？
- false positive 和 false negative 是否能在中间特征空间里提前暴露？
- boundary 和 hard background 的表征是否会随着训练逐渐分开？

SegEvo 希望把这些问题变成可记录、可回放、可比较的训练过程证据。

## 快速开始

```bash
git clone https://github.com/DZW131/SegEvo.git
cd SegEvo
pip install -e ".[dashboard]"

segevo-demo --out runs/demo
segevo-dashboard --run runs/demo --host 0.0.0.0 --port 7860
```

如果是在服务器上运行，可以在本地做 SSH 端口转发：

```bash
ssh -L 7860:localhost:7860 user@server
```

然后在本地浏览器打开：

```text
http://localhost:7860
```

如果是在比较老的 Linux 科研服务器上安装，建议先升级 pip 相关工具，确保能优先使用预编译 wheel：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dashboard]"
```

SegEvo 已经把 dashboard 依赖里的 `pyarrow` 固定在 `<21`，这样可以避开部分老 glibc 服务器上新版 `pyarrow` 找不到兼容 wheel、转而源码编译并报 Rust/C++ 编译错误的问题。

当前 dashboard 包含三个主要页面：

- `Case Timeline`：查看指定病例和 epoch 的原图、GT、prediction、FP/FN error map、指标曲线和 feature sample 计数。
- `Feature Space`：对采样到的中间层特征做稳定 3D PCA 投影，按 foreground、boundary、hard background、FP、FN 上色。
- `Boundary Learning`：查看 boundary Dice、surface Dice、HD95，以及边界特征与 foreground / hard background 的分离趋势。

## PyTorch U-Net 示例

项目内置了一个真实可运行的 PyTorch tiny U-Net 训练示例。它使用合成的 lesion-like 2D 分割数据，规模很小，CPU 也能跑，目的是展示 SegEvo 如何接入一个普通训练循环。

```bash
pip install -e ".[torch,dashboard]"
python examples/pytorch_unet_training.py --epochs 3 --run-dir runs/pytorch_unet
segevo-dashboard --run runs/pytorch_unet --host 0.0.0.0 --port 7860
```

SegEvo 相关的接入代码核心只有几步：

```python
from segevo import SegEvoLogger

logger = SegEvoLogger(
    run_dir="runs/pytorch_unet",
    manifest={
        "project": "PyTorch tiny U-Net example",
        "task": "binary_2d_synthetic_lesion_segmentation",
        "framework": "pytorch",
        "model": "TinyUNet",
        "classes": ["background", "lesion"],
    },
)

logger.attach(model, layers=["enc2", "bottleneck", "dec1"])

# 在验证或 probe-case logging 阶段记录
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

## 最小训练接入方式

第一版最稳定的 API 是显式记录。只要你的训练循环能拿到 image、GT mask 和 prediction，就可以接入：

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

如果需要记录 PyTorch 中间层特征摘要和采样向量：

```python
logger.attach(model, layers=["encoder.3", "bottleneck", "decoder.2"])
```

SegEvo 默认只保存紧凑的统计信息和采样特征，不会直接保存完整 feature volume。这样更适合医学 3D 数据，避免 artifact 过大。

## 实验产物格式

SegEvo dashboard 只依赖 run 目录，不依赖训练脚本本身。典型目录结构如下：

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
          uncertainty.npy
          features.npz
        0005/
          pred.npy
          error.npy
```

`error.npy` 使用紧凑整数标签：

- `0`：true background
- `1`：true positive
- `2`：false positive
- `3`：false negative

`features.npz` 保存紧凑层特征信息。以 `bottleneck` 为例：

- `bottleneck__summary`：该层激活的 mean、std、min、max。
- `bottleneck__samples`：采样特征向量，形状为 `[N, C]`。
- `bottleneck__sample_region_ids`：每个采样向量对应的区域 ID。
- `bottleneck__sample_coords`：采样点在 feature map 上的坐标。
- `feature_region_names`：区域 ID 对应的名称。

当前支持五类特征采样区域：

- `foreground`
- `boundary`
- `hard_background`
- `false_positive`
- `false_negative`

## 当前能力

- 统一 run 目录格式：`manifest.json`、`metrics.csv`、per-case epoch artifacts。
- 支持 2D 和 3D binary segmentation mask 的显式 logging。
- Dice、volume error、boundary Dice、HD95、surface Dice 等指标。
- Synthetic demo run 生成器。
- PyTorch forward hook，用于记录指定层的激活摘要和采样特征。
- 五类区域 feature sampling：foreground、boundary、hard background、FP、FN。
- `Case Timeline` 页面：查看病例随 epoch 的预测和错误演化。
- `Feature Space` 页面：跨 case / epoch 的稳定 3D PCA 特征空间投影。
- `Boundary Learning` 页面：边界指标和边界特征分离趋势。
- GitHub Actions CI：自动运行 lint、unit tests 和 PyTorch smoke test。

## 开发

```bash
pip install -e ".[dashboard,dev]"
pytest
ruff check .
```

如果还需要跑 PyTorch 示例：

```bash
pip install -e ".[torch,dashboard,dev]"
python examples/pytorch_unet_training.py --epochs 3 --run-dir runs/pytorch_unet
```

每次 push 和 pull request 都会通过 GitHub Actions 自动运行基础检查。

## 项目状态

当前版本可以视为 `v0.1 MVP`：已经具备可安装、可接入 PyTorch 训练、可生成 artifact、可启动 dashboard、可自动测试的基本开源项目形态。

后续计划包括：

- UMAP feature-space replay。
- Failure Explorer，用于定位持续难学、反复遗忘、后期仍错误的病例。
- 更多真实 PyTorch 2D / 3D 医学分割示例。
- 更完善的 uncertainty、attention / Grad-CAM artifact 约定。
- 文档截图、GIF、真实实验案例和 release tag。
