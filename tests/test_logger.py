import csv
import numpy as np

from segevo import SegEvoLogger


def test_logger_writes_case_artifacts(tmp_path):
    logger = SegEvoLogger(tmp_path)
    image = np.zeros((8, 8), dtype=np.float32)
    gt = np.zeros((8, 8), dtype=np.uint8)
    pred = np.zeros((8, 8), dtype=np.uint8)
    gt[2:5, 2:5] = 1
    pred[3:6, 3:6] = 1

    logger.log_case(epoch=0, case_id="case/001", image=image, gt=gt, pred=pred)

    case_dir = tmp_path / "cases" / "case_001"
    assert (case_dir / "image.npy").exists()
    assert (case_dir / "gt.npy").exists()
    assert (case_dir / "epochs" / "0000" / "pred.npy").exists()
    assert (case_dir / "epochs" / "0000" / "error.npy").exists()

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert metrics[0]["case_id"] == "case/001"
    assert 0.0 < float(metrics[0]["dice"]) < 1.0


def test_logger_samples_spatial_features(tmp_path):
    logger = SegEvoLogger(tmp_path, max_feature_samples_per_region=4)
    image = np.zeros((16, 16), dtype=np.float32)
    gt = np.zeros((16, 16), dtype=np.uint8)
    pred = np.zeros((16, 16), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    pred[5:13, 5:13] = 1
    feature = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)

    logger.log_case(
        epoch=0,
        case_id="case_001",
        image=image,
        gt=gt,
        pred=pred,
        features={"manual_layer": feature},
    )

    feature_path = tmp_path / "cases" / "case_001" / "epochs" / "0000" / "features.npz"
    with np.load(feature_path) as features:
        assert "manual_layer__summary" in features.files
        assert "manual_layer__samples" in features.files
        assert "manual_layer__sample_region_ids" in features.files
        assert "manual_layer__sample_coords" in features.files
        assert features["manual_layer__samples"].shape[1] == 3
