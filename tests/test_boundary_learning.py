import math

import numpy as np

from segevo.boundary_learning import boundary_learning_records, feature_boundary_separation
from segevo.logger import SegEvoLogger


def test_feature_boundary_separation_uses_region_centroids():
    region_names = ("foreground", "boundary", "hard_background")
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
            [3.0, 0.0],
            [3.2, 0.0],
        ],
        dtype=np.float32,
    )
    region_ids = np.asarray([1, 1, 0, 0, 2, 2], dtype=np.int16)

    values = feature_boundary_separation(features, region_ids, region_names)

    assert values["boundary_to_foreground"] > 0
    assert values["boundary_to_hard_background"] > values["boundary_to_foreground"]
    assert values["boundary_hard_margin"] > 1


def test_boundary_learning_records_from_logged_run(tmp_path):
    logger = SegEvoLogger(tmp_path, max_feature_samples_per_region=6)
    image = np.zeros((16, 16), dtype=np.float32)
    gt = np.zeros((16, 16), dtype=np.uint8)
    pred = np.zeros((16, 16), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    pred[5:13, 5:13] = 1
    feature = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)

    logger.log_case(0, "case_001", image, gt, pred, features={"manual_layer": feature})
    logger.log_case(1, "case_001", image, gt, gt, features={"manual_layer": feature + 1})

    records = boundary_learning_records(tmp_path, layers=["manual_layer"])

    assert len(records) == 2
    assert {record["epoch"] for record in records} == {0, 1}
    assert all(record["layer"] == "manual_layer" for record in records)
    assert all(0 <= record["boundary_dice"] <= 1 for record in records)
    assert records[1]["boundary_dice"] >= records[0]["boundary_dice"]
    assert any(math.isfinite(record["boundary_to_hard_background"]) for record in records)

