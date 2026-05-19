"""Low-intrusion logging API for segmentation training loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from segevo.artifacts import (
    append_metrics,
    case_dir,
    epoch_dir,
    save_array,
    to_numpy,
    write_manifest,
)
from segevo.metrics import error_map, summarize_binary_segmentation


class SegEvoLogger:
    """Write SegEvo artifacts from any segmentation training loop."""

    def __init__(
        self,
        run_dir: str | Path,
        manifest: Mapping[str, Any] | None = None,
        spacing: tuple[float, ...] | None = None,
        surface_tolerance: float = 1.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.spacing = spacing
        self.surface_tolerance = surface_tolerance
        self._hooks: list[Any] = []
        self._activations: dict[str, np.ndarray] = {}

        write_manifest(
            self.run_dir,
            {
                "project": "SegEvo run",
                "task": "binary_segmentation",
                **dict(manifest or {}),
            },
        )

    def attach(self, model: Any, layers: list[str] | tuple[str, ...]) -> None:
        """Attach PyTorch forward hooks and store compact layer summaries.

        The method is optional. It only depends on PyTorch-like ``named_modules`` and
        ``register_forward_hook`` APIs, so training code can ignore it when hooks are
        inconvenient.
        """
        modules = dict(model.named_modules())
        missing = [name for name in layers if name not in modules]
        if missing:
            raise ValueError(f"Unknown layer names: {missing}")

        for name in layers:
            handle = modules[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def log_case(
        self,
        epoch: int,
        case_id: str,
        image: Any,
        gt: Any,
        pred: Any,
        metrics: Mapping[str, Any] | None = None,
        features: Mapping[str, Any] | None = None,
        uncertainty: Any | None = None,
    ) -> None:
        """Log one probe case at one epoch."""
        image_np = _squeeze_channel(to_numpy(image))
        gt_np = _squeeze_channel(to_numpy(gt))
        pred_np = _squeeze_channel(to_numpy(pred))
        pred_np = _as_mask(pred_np)
        gt_np = _as_mask(gt_np)

        root = case_dir(self.run_dir, case_id)
        image_path = root / "image.npy"
        gt_path = root / "gt.npy"
        if not image_path.exists():
            save_array(image_path, image_np)
        if not gt_path.exists():
            save_array(gt_path, gt_np)

        epoch_path = epoch_dir(self.run_dir, case_id, epoch)
        save_array(epoch_path / "pred.npy", pred_np)
        save_array(epoch_path / "error.npy", error_map(pred_np, gt_np))
        if uncertainty is not None:
            save_array(epoch_path / "uncertainty.npy", _squeeze_channel(to_numpy(uncertainty)))

        feature_payload = {
            **self._activation_summaries(),
            **_feature_summaries(features or {}),
        }
        if feature_payload:
            np.savez_compressed(epoch_path / "features.npz", **feature_payload)

        computed = summarize_binary_segmentation(
            pred_np,
            gt_np,
            spacing=self.spacing,
            surface_tolerance=self.surface_tolerance,
        )
        row = {
            "epoch": int(epoch),
            "case_id": str(case_id),
            **computed,
            **dict(metrics or {}),
        }
        append_metrics(self.run_dir, row)

    def _make_hook(self, name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._activations[name] = to_numpy(output)

        return hook

    def _activation_summaries(self) -> dict[str, np.ndarray]:
        return {
            f"{name}__summary": summarize_feature(value)
            for name, value in self._activations.items()
        }


def summarize_feature(value: Any) -> np.ndarray:
    array = np.asarray(to_numpy(value), dtype=np.float32)
    if array.size == 0:
        return np.asarray([0, 0, 0, 0], dtype=np.float32)
    return np.asarray(
        [
            float(array.mean()),
            float(array.std()),
            float(array.min()),
            float(array.max()),
        ],
        dtype=np.float32,
    )


def _feature_summaries(features: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        f"{name}__summary": summarize_feature(value)
        for name, value in features.items()
    }


def _squeeze_channel(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array


def _as_mask(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == bool:
        return array.astype(np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        return (array >= 0.5).astype(np.uint8)
    return (array > 0).astype(np.uint8)

