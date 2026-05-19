# Pathology U-Net Integration

This guide shows how to connect SegEvo to the popular
[`milesial/Pytorch-UNet`](https://github.com/milesial/Pytorch-UNet) training script
for pathology tile segmentation.

The example assumes a server layout like:

```text
/home/data/jingkun/duyanhong/workspace/
  SegEvo/
  pathology_unet/
    train.py

/home/data/jingkun/duyanhong/dataspace/testdata/kmms_training/kmms_training/
  images/
    TCGA-18-5592-01Z-00-DX1.tif
  masks/
    TCGA-18-5592-01Z-00-DX1.png
```

## 1. Check Mask Names

`milesial/Pytorch-UNet` matches masks by filename stem. For an image named:

```text
TCGA-18-5592-01Z-00-DX1.tif
```

the matching mask should be:

```text
TCGA-18-5592-01Z-00-DX1.png
```

not:

```text
TCGA-18-5592-01Z-00-DX1 .png
```

To find and fix masks with a trailing space before `.png`:

```bash
cd /home/data/jingkun/duyanhong/dataspace/testdata/kmms_training/kmms_training/masks
find . -name "* .png" -print

python - <<'PY'
from pathlib import Path

for path in Path(".").glob("* .png"):
    target = path.with_name(path.name.replace(" .png", ".png"))
    if target.exists():
        raise SystemExit(f"Refuse to overwrite existing file: {target}")
    print(f"{path} -> {target}")
    path.rename(target)
PY
```

## 2. Link Data Into Pytorch-UNet

The original training script expects `./data/imgs` and `./data/masks`.

```bash
cd /home/data/jingkun/duyanhong/workspace/pathology_unet
mkdir -p data

ln -sfn /home/data/jingkun/duyanhong/dataspace/testdata/kmms_training/kmms_training/images data/imgs
ln -sfn /home/data/jingkun/duyanhong/dataspace/testdata/kmms_training/kmms_training/masks data/masks

ls data/imgs | head
ls data/masks | head
```

## 3. Install SegEvo

If PyTorch is already installed in the active conda environment, install only the
dashboard extra:

```bash
conda activate segevo_test
cd /home/data/jingkun/duyanhong/workspace/SegEvo
git pull
python -m pip install -e ".[dashboard]"
```

## 4. Patch `train.py`

Edit:

```text
/home/data/jingkun/duyanhong/workspace/pathology_unet/train.py
```

Add this import block near the existing imports:

```python
from segevo import SegEvoLogger
```

Add these helper functions before `train_model`:

```python
def chw_to_hwc(image):
    image = image.detach().float().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


@torch.inference_mode()
def log_segevo_probes(model, probe_loader, device, logger, epoch, amp, max_cases):
    model.eval()
    logged = 0
    for batch in probe_loader:
        images = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        true_masks = batch["mask"].to(device=device, dtype=torch.long)

        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            logits = model(images)
            if model.n_classes == 1:
                probs = torch.sigmoid(logits).squeeze(1)
                preds = (probs >= 0.5).long()
                uncertainty = 1.0 - torch.abs(probs - 0.5) * 2.0
            else:
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                uncertainty = 1.0 - probs.max(dim=1).values

        for item_index in range(images.shape[0]):
            case_id = f"probe_{logged:03d}"
            logger.log_case(
                epoch=epoch,
                case_id=case_id,
                image=chw_to_hwc(images[item_index]),
                gt=true_masks[item_index].detach().cpu().numpy(),
                pred=preds[item_index].detach().cpu().numpy(),
                uncertainty=uncertainty[item_index].detach().cpu().numpy(),
            )
            logged += 1
            if logged >= max_cases:
                model.train()
                return
    model.train()
```

Extend `train_model` arguments:

```python
        segevo_run: str | None = None,
        segevo_interval: int = 1,
        segevo_probes: int = 4,
```

After `val_loader` is created, add:

```python
    probe_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    segevo_logger = None
    if segevo_run:
        segevo_logger = SegEvoLogger(
            run_dir=segevo_run,
            manifest={
                "project": "Pathology U-Net",
                "task": "pathology_tile_segmentation",
                "classes": [str(value) for value in dataset.mask_values],
            },
            max_feature_samples_per_region=128,
        )
        segevo_logger.attach(model, layers=["inc", "down2", "down4", "up1"])
```

At the end of each epoch, before checkpoint saving, add:

```python
        if segevo_logger is not None and epoch % segevo_interval == 0:
            log_segevo_probes(
                model=model,
                probe_loader=probe_loader,
                device=device,
                logger=segevo_logger,
                epoch=epoch,
                amp=amp,
                max_cases=segevo_probes,
            )
```

Add CLI arguments in `get_args()`:

```python
    parser.add_argument("--segevo-run", type=str, default=None, help="SegEvo run directory")
    parser.add_argument("--segevo-interval", type=int, default=1, help="Log SegEvo probes every N epochs")
    parser.add_argument("--segevo-probes", type=int, default=4, help="Number of fixed validation probes")
```

Pass them into `train_model(...)`:

```python
            segevo_run=args.segevo_run,
            segevo_interval=args.segevo_interval,
            segevo_probes=args.segevo_probes,
```

## 5. Train And View

Start with a small run:

```bash
cd /home/data/jingkun/duyanhong/workspace/pathology_unet
conda activate segevo_test

python train.py \
  --epochs 3 \
  --batch-size 2 \
  --scale 0.5 \
  --classes 2 \
  --amp \
  --segevo-run /home/data/jingkun/duyanhong/workspace/SegEvo/runs/pathology_unet \
  --segevo-interval 1 \
  --segevo-probes 4
```

Then open the dashboard:

```bash
cd /home/data/jingkun/duyanhong/workspace/SegEvo
segevo-dashboard --run runs/pathology_unet --host 0.0.0.0 --port 7860
```

Use SSH forwarding if needed:

```bash
ssh -L 7860:localhost:7860 jingkun@10.16.111.73
```

Then open:

```text
http://localhost:7860
```

