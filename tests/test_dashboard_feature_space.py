import numpy as np
import pandas as pd

from segevo.dashboard import (
    _centroid_dataframe,
    _default_regions_for_preset,
    _feature_coord_to_image_coord,
    _feature_separation_metrics,
    _nearest_epoch,
    _selected_point_id,
)


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


def test_region_presets_and_epoch_sync_helpers():
    all_regions = ["foreground", "boundary", "hard_background", "false_positive"]

    assert _default_regions_for_preset(all_regions, "Core regions") == [
        "foreground",
        "boundary",
        "hard_background",
    ]
    assert _default_regions_for_preset(all_regions, "Errors only (FP/FN)") == [
        "false_positive"
    ]
    assert _nearest_epoch([0, 2, 4, 8], requested_epoch=5) == 4


def test_feature_point_selection_and_coordinate_mapping():
    event = {"selection": {"points": [{"customdata": [17, "case_001", 2]}]}}

    assert _selected_point_id(event) == 17
    assert _feature_coord_to_image_coord(
        feature_coord=(2, 4),
        spatial_shape=(10, 10),
        image_shape=(100, 200, 3),
    ) == (25, 90)
