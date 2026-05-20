import numpy as np
import pandas as pd

from segevo.dashboard import (
    _boundary_diagnosis,
    _error_balance_rows,
    _error_balance_sentence,
    _metric_readout_rows,
)


def test_metric_readout_rows_describe_improvement_directions():
    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "dice": [0.2, 0.7],
            "hd95": [12.0, 5.0],
            "volume_error": [0.5, 0.1],
        }
    )

    rows = _metric_readout_rows(df, ("dice", "hd95", "volume_error"), focus_epoch=1)
    statuses = {row["metric"]: row["status"] for row in rows}

    assert statuses["dice"] == "improved vs first epoch"
    assert statuses["hd95"] == "improved vs first epoch"
    assert statuses["volume_error"] == "closer to target than first epoch"


def test_error_balance_rows_explain_fp_fn_dominance():
    err = np.zeros((6, 6), dtype=np.uint8)
    err[0:2, :] = 1
    err[2:5, :] = 2
    err[5, 0:2] = 3

    rows = _error_balance_rows(err)
    values = {row["signal"]: row["value"] for row in rows}

    assert values["true_positive"] == 12
    assert values["false_positive"] == 18
    assert values["false_negative"] == 2
    assert "FP-dominant" in _error_balance_sentence(rows)


def test_boundary_diagnosis_reports_metric_trends():
    rows = [
        {"metric": "boundary_dice", "status": "improved vs first epoch"},
        {"metric": "hd95", "status": "improved vs first epoch"},
        {"metric": "boundary_to_hard_background", "status": "improved vs first epoch"},
    ]

    diagnosis = _boundary_diagnosis(rows)

    assert any("Boundary overlap improved" in item for item in diagnosis)
    assert any("HD95 went down" in item for item in diagnosis)
    assert any("separating from hard background" in item for item in diagnosis)
