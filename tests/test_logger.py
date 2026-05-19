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
