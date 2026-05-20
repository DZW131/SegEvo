# SegEvo

[![CI](https://github.com/DZW131/SegEvo/actions/workflows/ci.yml/badge.svg)](https://github.com/DZW131/SegEvo/actions/workflows/ci.yml)

**语言:** [English](README.md) | 简体中文

SegEvo 是一个轻量、模型无关的医学影像分割训练观测工具。它会在训练过程中反复记录一小组固定
probe 病例，并把这些记录变成可交互 dashboard，帮助你理解模型是如何学习的、错在哪里、边界
是否变稳、中间特征表征是否逐渐分开。

SegEvo 不是新的训练框架，而是一个可以接到现有训练代码上的观测层。它不仅适用于 U-Net，也可以
接入 nnU-Net 风格代码、UNETR/SwinUNETR、DeepLab 类模型、MedNeXt 类模型、WSI tile 分割模型，
以及各种自定义 CNN/Transformer 分割模型。只要你的训练循环能提供 image、ground truth、
prediction，以及可选的中间层特征，就可以使用 SegEvo。

## 效果演示视频

[观看 SegEvo dashboard 演示](docs/assets/segevo-dashboard-demo.mp4)

这个录屏依次展示了三个主要页面：病例级训练时间线、3D feature space 演化，以及边界学习诊断。

## 为什么需要 SegEvo

最终 Dice 或 HD95 只能告诉你训练结束后结果怎么样，但经常无法解释模型是怎么变成这样的。

SegEvo 关注训练过程本身，帮助回答这些问题：

- 某个固定 probe 病例是在稳定变好，还是训练过程中反复退化？
- 剩余错误主要是假阳性、多分了，还是假阴性、漏分了，或者主要是边界偏移？
- Dice 变好是否掩盖了边界问题？
- foreground、boundary、hard background、FP、FN 的中间特征是否随着训练逐渐分开？
- 哪个网络层最能解释当前模型的学习状态？
- 模型是真的学到了稳定表征，还是只学到了容易区域？

这些问题在医学影像分割里很常见：小目标、模糊边界、hard negative 组织、扫描域偏移、验证病例少，
都会让单一最终指标不够可靠。

## 效果展示与页面解读

运行内置 demo 并打开 dashboard：

```bash
segevo-demo --out runs/demo
segevo-dashboard --run runs/demo --host 0.0.0.0 --port 7860
```

dashboard 目前有三个主要页面：

| 页面 | 看到什么 | 能读出什么训练信息 |
| --- | --- | --- |
| `Case Timeline` | Image + GT、Image + Prediction、FP/FN Error Map、指标曲线、当前 epoch 训练读数。 | 这个固定病例是否在变好，错误主要来自 FP 还是 FN，指标变好是否真的对应视觉变好。 |
| `Feature Space` | 中间层采样特征的稳定 3D PCA 投影、区域预设、只看 FP/FN、中心轨迹、epoch 播放、convex hull / density surface、点选定位、HTML 导出。 | 模型内部是否把 foreground、boundary、hard background、FP、FN 组织成更清楚的特征簇。 |
| `Boundary Learning` | boundary Dice、surface Dice、HD95、边界特征分离趋势、边界训练读数。 | 模型是否真的学到了可用边界，而不是只有粗略 mask 重叠变好。 |

每个页面顶部都有中英文说明，包含控件怎么用、图怎么看、以及可以从图里推断哪些训练状态。

## 安装

SegEvo 需要 Python 3.10 或更新版本。

### 方式 A：virtualenv

```bash
git clone https://github.com/DZW131/SegEvo.git
cd SegEvo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dashboard]"
```

### 方式 B：conda

```bash
git clone https://github.com/DZW131/SegEvo.git
cd SegEvo
conda create -n segevo python=3.10 -y
conda activate segevo
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dashboard]"
```

### PyTorch 项目

如果你的训练环境里已经装好了 PyTorch，只需要安装 dashboard 依赖：

```bash
python -m pip install -e ".[dashboard]"
```

如果要运行项目自带的 PyTorch 示例：

```bash
python -m pip install -e ".[torch,dashboard]"
```

GPU 训练时，请先根据自己的 CUDA/runtime 环境安装匹配的 PyTorch，然后在同一个环境里安装 SegEvo。

### 老 Linux 服务器

在较老的科研服务器上，建议先升级 pip 相关工具，让 pip 尽量使用预编译 wheel：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dashboard]"
```

SegEvo 对 dashboard 依赖里的 `pyarrow` 做了版本约束，目的是减少老 glibc 系统上找不到新 wheel、
转而源码编译导致失败的概率。

## 快速开始

```bash
git clone https://github.com/DZW131/SegEvo.git
cd SegEvo
python -m pip install -e ".[dashboard]"

segevo-demo --out runs/demo
segevo-dashboard --run runs/demo --host 0.0.0.0 --port 7860
```

本地浏览器打开：

```text
http://localhost:7860
```

如果 dashboard 跑在远程服务器上，可以在本地做端口转发：

```bash
ssh -L 7860:localhost:7860 user@server
```

然后在本地浏览器打开 `http://localhost:7860`。

## 在自己的训练项目中使用

推荐流程：

1. 从验证集里选一小组固定 probe 病例。
2. 每隔若干 epoch 记录同一组 probe 病例。
3. 如果需要 feature space 和 boundary feature 分析，就给 PyTorch 模型 attach 若干中间层。
4. 训练过程中或训练结束后打开 dashboard 查看。

### 最小显式 logging

只要你能拿到 image、GT mask 和 prediction，就可以接入这个方式。它不要求你的模型一定是 U-Net。

```python
from segevo import SegEvoLogger

logger = SegEvoLogger(
    run_dir="runs/my_segmentation_model",
    manifest={
        "project": "my_segmentation_model",
        "task": "binary_lesion_segmentation",
        "model": "MyModel",
        "classes": ["background", "lesion"],
        "spacing": [1.0, 1.0, 1.0],  # 可选，用于 HD95 / surface metrics
    },
)

for epoch in range(num_epochs):
    train_one_epoch(...)
    validation_metrics = validate(...)

    if epoch % 5 == 0:
        for case in fixed_probe_cases:
            image = case["image"]
            gt = case["mask"]
            pred = infer_binary_mask(model, image)

            logger.log_case(
                epoch=epoch,
                case_id=case["case_id"],
                image=image,
                gt=gt,
                pred=pred,
                metrics={"val_dice": validation_metrics["dice"]},
            )
```

### PyTorch 中间层 feature hook

对于 PyTorch 模型，可以在推理前 attach 指定层。SegEvo 会记录紧凑的激活摘要和区域采样特征。

```python
logger.attach(
    model,
    layers=[
        "encoder.stage2",
        "bottleneck",
        "decoder.stage1",
    ],
)
```

层名可以这样查看：

```python
for name, _module in model.named_modules():
    print(name)
```

在 `logger.log_case(...)` 时，SegEvo 会从以下区域采样 feature：

- `foreground`
- `boundary`
- `hard_background`
- `false_positive`
- `false_negative`

这些采样会驱动 `Feature Space` 和 `Boundary Learning` 页面。

### PyTorch 示例

项目内置了一个 CPU 也能跑的 tiny U-Net 示例，用于展示普通 PyTorch 训练循环如何接入 SegEvo：

```bash
python -m pip install -e ".[torch,dashboard]"
python examples/pytorch_unet_training.py --epochs 3 --run-dir runs/pytorch_unet
segevo-dashboard --run runs/pytorch_unet --host 0.0.0.0 --port 7860
```

如果要把 SegEvo 接到真实 PyTorch 分割训练脚本里，可以参考：

[docs/pytorch_integration.md](docs/pytorch_integration.md)

## 数据和模型适配范围

当前版本重点支持：

- 2D 和 3D binary segmentation mask。
- CT、MR、超声、病理 tile、显微图像 tile，或任何能保存为 NumPy-compatible array 的图像数据。
- 任意带有 `named_modules()` 和 forward hook 的 PyTorch 分割模型。
- 非 PyTorch 流程也可以通过显式 `logger.log_case(...)` 使用。

对于 multiclass segmentation，目前建议先按重要类别做 foreground-vs-background 的 one-vs-rest
记录，例如先记录病灶类、器官类或临床最关心的类别。原生 multiclass dashboard 是后续扩展方向。

## Artifact 目录结构

SegEvo 会在本地写出一个 run 目录。dashboard 只读取这个目录，不需要重新 import 或运行你的训练代码。

```text
runs/my_segmentation_model/
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
          features.npz
```

`error.npy` 使用紧凑整数标签：

- `0`：true background
- `1`：true positive
- `2`：false positive
- `3`：false negative

`features.npz` 保存紧凑层特征信息。以 `bottleneck` 为例：

- `bottleneck__summary`：激活 mean、std、min、max。
- `bottleneck__samples`：采样特征向量，形状为 `[N, C]`。
- `bottleneck__sample_region_ids`：每个采样向量对应的区域 ID。
- `bottleneck__sample_coords`：采样点在 feature map 上的坐标。
- `bottleneck__sample_spatial_shape`：被采样 feature map 的空间尺寸。
- `feature_region_names`：区域 ID 对应的名称。

SegEvo 默认不保存完整 dense feature volume，因为 3D 医学影像很容易产生巨大的 artifact。

## 隐私和数据安全

- SegEvo 会把 artifact 写到你指定的本地 run 目录。
- 包本身不包含数据上传逻辑，也不依赖外部服务。
- dashboard 只读取本地 artifact 并用 Streamlit 渲染。
- 提 issue 时请不要上传患者图像、原始私有数据、token、服务器 IP、私有路径等信息。
  如果需要复现问题，请使用合成数据或脱敏后的最小示例。

## 仓库内容

```text
src/segevo/                 # package 源码
examples/minimal_logging.py # 框架无关的最小 logging 示例
examples/pytorch_unet_training.py
docs/pytorch_integration.md # 通用 PyTorch 接入指南
tests/                      # 单元测试和 smoke tests
.github/workflows/ci.yml    # lint/test CI
```

生成的 runs、本地环境、缓存、日志和 `.env` 文件已经被 `.gitignore` 忽略。

## 开发

```bash
python -m pip install -e ".[dashboard,dev]"
python -m ruff check .
python -m pytest -q
```

本地运行 PyTorch smoke example：

```bash
python -m pip install -e ".[torch,dashboard,dev]"
python examples/pytorch_unet_training.py --epochs 1 --run-dir runs/local_smoke
```

每次 push 和 pull request 都会通过 GitHub Actions 运行 lint、compile、unit tests 和 tiny PyTorch
U-Net smoke test。

## 贡献和 Issues

欢迎提交 issue 和 pull request。一个有帮助的 bug report 通常包括：

- 操作系统和 Python 版本。
- SegEvo commit 或 release 版本。
- 安装命令。
- dashboard 启动命令。
- 最小化、合成或脱敏后的 run 目录结构。
- 报错 traceback 或截图，注意移除隐私信息。

适合贡献的方向：

- 更多 2D / 3D 医学影像分割项目接入示例。
- 原生 multiclass segmentation 支持。
- UMAP 等更多 feature-space 投影方式。
- Failure Explorer，用于定位不稳定、反复遗忘、后期仍错误的 probe 病例。
- 更适合论文和报告的导出格式。

## 当前限制

SegEvo 目前仍是 MVP 阶段的科研工具，主要限制包括：

- binary segmentation 是当前第一优先支持路径。
- feature-space 图是诊断性投影，不是形式化统计证明。
- probe-case logging 是采样式记录，不适合保存每个 batch 或完整 dense feature volume。
- 大型实验建议只记录少量固定代表性病例。

## License

MIT License. See [LICENSE](LICENSE).
