"""Filesystem helpers for SegEvo run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


METRICS_COLUMNS = [
    "epoch",
    "case_id",
    "dice",
    "boundary_dice",
    "hd95",
    "surface_dice",
    "volume_error",
]


def ensure_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "cases").mkdir(exist_ok=True)
    return path


def write_manifest(run_dir: str | Path, manifest: Mapping[str, Any]) -> None:
    path = ensure_run_dir(run_dir) / "manifest.json"
    payload = {
        "schema_version": "0.1",
        **dict(manifest),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return {"schema_version": "0.1"}
    return json.loads(path.read_text(encoding="utf-8"))


def case_dir(run_dir: str | Path, case_id: str) -> Path:
    path = ensure_run_dir(run_dir) / "cases" / sanitize_id(case_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def epoch_dir(run_dir: str | Path, case_id: str, epoch: int) -> Path:
    path = case_dir(run_dir, case_id) / "epochs" / f"{epoch:04d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return safe or "case"


def save_array(path: str | Path, array: Any) -> None:
    np.save(Path(path), to_numpy(array), allow_pickle=False)


def load_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def append_metrics(run_dir: str | Path, row: Mapping[str, Any]) -> None:
    path = ensure_run_dir(run_dir) / "metrics.csv"
    fields = list(dict.fromkeys([*METRICS_COLUMNS, *row.keys()]))

    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        merged_fields = list(dict.fromkeys([*existing_fields, *fields]))
        existing_rows.append({key: _csv_value(value) for key, value in row.items()})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_fields)
            writer.writeheader()
            writer.writerows(existing_rows)
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: _csv_value(value) for key, value in row.items()})


def list_cases(run_dir: str | Path) -> list[str]:
    root = Path(run_dir) / "cases"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def list_epochs(run_dir: str | Path, case_id: str) -> list[int]:
    root = Path(run_dir) / "cases" / sanitize_id(case_id) / "epochs"
    if not root.exists():
        return []
    epochs: list[int] = []
    for path in root.iterdir():
        if path.is_dir() and path.name.isdigit():
            epochs.append(int(path.name))
    return sorted(epochs)


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
