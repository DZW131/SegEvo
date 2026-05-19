import numpy as np

from segevo.feature_space import (
    available_feature_layers,
    load_feature_space,
    pca_2d,
    pca_3d,
    project_feature_space,
)
from segevo.logger import SegEvoLogger


def test_pca_2d_projects_with_stable_shape():
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    x, y, ratios = pca_2d(features)
    assert x.shape == (4,)
    assert y.shape == (4,)
    assert ratios[0] > 0.99


def test_pca_3d_projects_with_stable_shape():
    features = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    x, y, z, ratios = pca_3d(features)
    assert x.shape == y.shape == z.shape == (4,)
    assert len(ratios) == 3
    assert ratios[0] > 0.7


def test_load_and_project_feature_space_from_run(tmp_path):
    logger = SegEvoLogger(tmp_path, max_feature_samples_per_region=3)
    image = np.zeros((16, 16), dtype=np.float32)
    gt = np.zeros((16, 16), dtype=np.uint8)
    pred = np.zeros((16, 16), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    pred[5:13, 5:13] = 1
    feature = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)

    logger.log_case(0, "case_001", image, gt, pred, features={"manual_layer": feature})
    logger.log_case(1, "case_001", image, gt, gt, features={"manual_layer": feature + 1})

    assert available_feature_layers(tmp_path) == ["manual_layer"]

    space = load_feature_space(tmp_path, layer="manual_layer", max_points=20)
    assert space.features.shape[0] <= 20
    assert space.features.shape[1] == 3
    assert set(space.epochs.tolist()) == {0, 1}

    projection = project_feature_space(space)
    assert projection.x.shape == projection.y.shape == projection.z.shape == space.region_ids.shape
    assert projection.layer == "manual_layer"
