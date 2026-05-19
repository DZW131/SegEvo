"""PyTorch U-Net training example with SegEvo logging.

This is a small, self-contained example that mirrors a normal segmentation
training loop. The only SegEvo-specific lines are creating a logger, attaching
layers, and logging fixed probe cases every few epochs.

Run:
    pip install -e ".[torch]"
    python examples/pytorch_unet_training.py --epochs 3 --run-dir runs/pytorch_unet
    segevo-dashboard --run runs/pytorch_unet --port 7860
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from segevo import SegEvoLogger, dice_score


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.out(dec1)


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    image: torch.Tensor
    mask: torch.Tensor


class SyntheticLesionDataset(Dataset[SyntheticCase]):
    """Small deterministic 2D lesion-like segmentation dataset."""

    def __init__(self, count: int, image_size: int = 96, seed: int = 0) -> None:
        self.count = count
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> SyntheticCase:
        rng = np.random.default_rng(self.seed + index)
        size = self.image_size
        yy, xx = np.mgrid[:size, :size]

        center_y = rng.integers(size // 3, 2 * size // 3)
        center_x = rng.integers(size // 3, 2 * size // 3)
        radius_y = rng.integers(max(5, size // 8), max(6, size // 5))
        radius_x = rng.integers(max(6, size // 7), max(7, size // 4))
        lesion = (((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2 <= 1)

        distractor_y = rng.integers(size // 5, 4 * size // 5)
        distractor_x = rng.integers(size // 5, 4 * size // 5)
        distractor = (
            ((yy - distractor_y) / max(4, radius_y // 2)) ** 2
            + ((xx - distractor_x) / max(4, radius_x // 2)) ** 2
            <= 1
        )

        image = 0.16 * rng.normal(size=(size, size))
        image += 0.65 * lesion.astype(np.float32)
        image += 0.25 * distractor.astype(np.float32)
        image += 0.10 * np.sin(xx / 8.0) + 0.06 * np.cos(yy / 11.0)
        image = (image - image.mean()) / (image.std() + 1e-6)

        image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        mask_t = torch.from_numpy(lesion.astype(np.float32)).unsqueeze(0)
        return SyntheticCase(case_id=f"synthetic_{index:03d}", image=image_t, mask=mask_t)


def collate_cases(batch: list[SyntheticCase]) -> dict[str, object]:
    return {
        "case_id": [case.case_id for case in batch],
        "image": torch.stack([case.image for case in batch]),
        "mask": torch.stack([case.mask for case in batch]),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        image = batch["image"].to(device)  # type: ignore[union-attr]
        mask = batch["mask"].to(device)  # type: ignore[union-attr]
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = loss_fn(logits, mask)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def log_probe_cases(
    model: nn.Module,
    probe_cases: list[SyntheticCase],
    logger: SegEvoLogger,
    epoch: int,
    train_loss: float,
    device: torch.device,
) -> None:
    model.eval()
    for case in probe_cases:
        image = case.image.unsqueeze(0).to(device)
        logits = model(image)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (prob >= 0.5).astype(np.uint8)
        gt = case.mask[0].cpu().numpy().astype(np.uint8)
        uncertainty = (prob * (1.0 - prob)).astype(np.float32)

        logger.log_case(
            epoch=epoch,
            case_id=case.case_id,
            image=case.image[0].cpu().numpy(),
            gt=gt,
            pred=pred,
            uncertainty=uncertainty,
            metrics={
                "train_loss": train_loss,
                "probe_dice": dice_score(pred, gt),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny PyTorch U-Net with SegEvo logging.")
    parser.add_argument(
        "--run-dir",
        default="runs/pytorch_unet",
        help="SegEvo output run directory.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--log-every", type=int, default=1, help="Log probe cases every N epochs.")
    parser.add_argument(
        "--train-cases",
        type=int,
        default=48,
        help="Number of synthetic training cases.",
    )
    parser.add_argument("--probe-cases", type=int, default=3, help="Number of fixed probe cases.")
    parser.add_argument("--image-size", type=int, default=96, help="Synthetic image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    train_dataset = SyntheticLesionDataset(
        count=args.train_cases,
        image_size=args.image_size,
        seed=args.seed,
    )
    probe_dataset = SyntheticLesionDataset(
        count=args.probe_cases,
        image_size=args.image_size,
        seed=args.seed + 10_000,
    )
    train_loader: DataLoader[dict[str, object]] = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_cases,
    )

    model = TinyUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    logger = SegEvoLogger(
        run_dir=run_dir,
        manifest={
            "project": "PyTorch tiny U-Net example",
            "task": "binary_2d_synthetic_lesion_segmentation",
            "framework": "pytorch",
            "model": "TinyUNet",
            "classes": ["background", "lesion"],
            "logged_layers": ["enc2", "bottleneck", "dec1"],
            "spacing": [1.0, 1.0],
        },
        spacing=(1.0, 1.0),
    )
    logger.attach(model, layers=["enc2", "bottleneck", "dec1"])

    probe_cases = [probe_dataset[index] for index in range(len(probe_dataset))]
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            log_probe_cases(model, probe_cases, logger, epoch, train_loss, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f}")

    logger.close()
    print(f"SegEvo run written to {run_dir}")
    print(f"Open it with: segevo-dashboard --run {run_dir} --port 7860")


if __name__ == "__main__":
    main()
