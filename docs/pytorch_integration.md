# Generic PyTorch Segmentation Integration

This guide shows how to connect SegEvo to an existing PyTorch segmentation training
script. The example is intentionally generic and does not assume a specific model
architecture. The same pattern can be used for U-Net, nnU-Net-style projects,
UNETR/SwinUNETR, DeepLab-like models, WSI tile segmentation models, or custom
segmentation networks.

## 1. Install SegEvo In The Training Environment

Use the same Python environment that runs your training script.

```bash
cd /path/to/SegEvo
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dashboard]"
```

If you want to run the bundled PyTorch example as well:

```bash
python -m pip install -e ".[torch,dashboard]"
```

If your project already manages PyTorch separately, install the PyTorch build that
matches your CUDA/runtime environment first, then install SegEvo.

## 2. Choose Fixed Probe Cases

SegEvo works best when the same small set of validation cases is logged repeatedly.
For example:

- 2 to 8 representative CT/MR volumes.
- Several pathology or microscopy tiles.
- A mixture of easy, difficult, small-object, boundary-heavy, and hard-negative cases.

Avoid logging every training batch. SegEvo is designed for focused training-process
observability, not bulk prediction storage.

## 3. Create A Logger

Add this near your training setup:

```python
from segevo import SegEvoLogger

segevo_logger = SegEvoLogger(
    run_dir="runs/my_model_segevo",
    manifest={
        "project": "my_model",
        "task": "binary_medical_segmentation",
        "model": type(model).__name__,
        "classes": ["background", "target"],
        "spacing": [1.0, 1.0, 1.0],  # optional for 3D physical-distance metrics
    },
    max_feature_samples_per_region=128,
)
```

`spacing` is optional but useful for HD95 and surface Dice when your voxel spacing is
known.

## 4. Attach Optional PyTorch Layers

If you want Feature Space and Boundary Learning to show intermediate representations,
attach a few model layers:

```python
for name, _module in model.named_modules():
    print(name)

segevo_logger.attach(
    model,
    layers=[
        "encoder.stage2",
        "bottleneck",
        "decoder.stage1",
    ],
)
```

Choose layers that represent different semantic depths:

- One encoder layer for texture or local structure.
- One bottleneck or transformer block for high-level semantics.
- One decoder layer for segmentation reconstruction.

If layer hooks are inconvenient, you can skip `attach(...)` and still use Case Timeline
and metric logging.

## 5. Convert Model Outputs To Binary Masks

SegEvo's current first-class path is binary segmentation. Typical conversions:

```python
# Binary logits: [B, 1, H, W] or [B, 1, D, H, W]
prob = torch.sigmoid(logits)
pred = (prob >= 0.5).long()
uncertainty = 1.0 - torch.abs(prob - 0.5) * 2.0

# Multiclass logits: [B, C, H, W] or [B, C, D, H, W]
class_pred = torch.softmax(logits, dim=1).argmax(dim=1)
target_class = 1
pred = (class_pred == target_class).long()
gt = (mask == target_class).long()
```

For multiclass tasks, log the most important target class first, or log multiple
one-vs-rest SegEvo runs.

## 6. Log Probe Cases

Call `log_case(...)` after inference on your fixed probe cases. A typical pattern is
to log every `N` epochs during validation.

```python
@torch.inference_mode()
def log_segevo_probes(model, probe_loader, device, logger, epoch, max_cases=4):
    model.eval()
    logged = 0

    for batch in probe_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        uncertainty = 1.0 - torch.abs(probs - 0.5) * 2.0

        for item_index in range(images.shape[0]):
            logger.log_case(
                epoch=epoch,
                case_id=str(batch["case_id"][item_index]),
                image=images[item_index].detach().cpu().numpy(),
                gt=masks[item_index].detach().cpu().numpy(),
                pred=preds[item_index].detach().cpu().numpy(),
                uncertainty=uncertainty[item_index].detach().cpu().numpy(),
            )
            logged += 1
            if logged >= max_cases:
                model.train()
                return

    model.train()
```

In your training loop:

```python
for epoch in range(num_epochs):
    train_one_epoch(...)
    val_metrics = validate(...)

    if epoch % segevo_interval == 0:
        log_segevo_probes(
            model=model,
            probe_loader=probe_loader,
            device=device,
            logger=segevo_logger,
            epoch=epoch,
            max_cases=4,
        )
```

## 7. Open The Dashboard

```bash
segevo-dashboard --run runs/my_model_segevo --host 0.0.0.0 --port 7860
```

For a remote server:

```bash
ssh -L 7860:localhost:7860 user@server
```

Then open `http://localhost:7860` locally.

## 8. Common Checks

If the dashboard says no cases were found:

- Make sure `logger.log_case(...)` has run at least once.
- Check that the `--run` path points to the same directory passed to `SegEvoLogger`.
- Check for `manifest.json`, `metrics.csv`, and `cases/` under the run directory.

If Feature Space is empty:

- Make sure `segevo_logger.attach(model, layers=[...])` was called before inference.
- Make sure the selected layer names exist in `model.named_modules()`.
- Make sure the forward pass used for logging actually runs after hooks are attached.

If masks look wrong:

- Confirm that `gt` and `pred` have the same spatial shape.
- Confirm that binary foreground is encoded as values greater than zero.
- For multiclass tasks, convert the target class to a binary mask before logging.

## Minimal Alternative Without Hooks

You can also log manually supplied features:

```python
logger.log_case(
    epoch=epoch,
    case_id=case_id,
    image=image_np,
    gt=gt_np,
    pred=pred_np,
    features={
        "custom_layer": feature_map_np,  # shape [C, H, W] or [C, D, H, W]
    },
)
```

This is useful for non-standard models, cached activations, or non-PyTorch workflows.
