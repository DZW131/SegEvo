import numpy as np
import pandas as pd

from segevo.dashboard import _centroid_dataframe, _feature_separation_metrics


def test_centroid_dataframe_summarizes_epoch_region_centers():
    df = pd.DataFrame(
        {
            "epoch": [1, 1, 1, 2],
            "region": ["boundary", "boundary", "foreground", "boundary"],
            "pc1": [0.0, 2.0, 5.0, 3.0],
            "pc2": [0.0, 0.0, 0.0, 1.0],
            "pc3": [0.0, 0.0, 0.0, 1.0],
        }
    )

    centroids = _centroid_dataframe(df)
    row = centroids[(centroids["epoch"] == 1) & (centroids["region"] == "boundary")].iloc[0]

    assert row["samples"] == 2
    assert np.isclose(row["pc1"], 1.0)


def test_feature_separation_metrics_report_boundary_and_error_distances():
    df = pd.DataFrame(
        {
            "epoch": [2, 2, 2, 2, 2],
            "region": [
                "boundary",
                "boundary",
                "foreground",
                "hard_background",
                "false_positive",
            ],
            "pc1": [0.0, 2.0, 4.0, -2.0, 5.0],
            "pc2": [0.0, 0.0, 0.0, 0.0, 0.0],
            "pc3": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    metrics = _feature_separation_metrics(df, epoch=2).set_index("metric")["value"]

    assert np.isclose(metrics["boundary_within"], 1.0)
    assert np.isclose(metrics["boundary_to_foreground"], 3.0)
    assert np.isclose(metrics["boundary_to_hard_background"], 3.0)
    assert np.isclose(metrics["boundary_hard_margin"], 2.0)
    assert np.isclose(metrics["fp_to_foreground"], 1.0)
