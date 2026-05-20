import numpy as np

from segevo.feature_space import (
    FeatureSpace,
    available_feature_layers,
    downsample_feature_space,
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
    assert space.spatial_shapes is not None
    assert space.spatial_shapes.shape == space.coords.shape
    assert set(space.epochs.tolist()) == {0, 1}

    projection = project_feature_space(space)
    assert projection.x.shape == projection.y.shape == projection.z.shape == space.region_ids.shape
    assert projection.layer == "manual_layer"
    assert projection.feature_spatial_shapes is not None
    assert projection.feature_spatial_shapes.shape == space.coords.shape


def test_downsample_feature_space_keeps_epoch_region_strata():
    epochs = np.repeat(np.asarray([0, 1, 2], dtype=np.int32), 20)
    region_ids = np.tile(np.repeat(np.asarray([0, 1], dtype=np.int16), 10), 3)
    features = np.arange(60 * 3, dtype=np.float32).reshape(60, 3)
    space = FeatureSpace(
        layer="manual_layer",
        features=features,
        region_ids=region_ids,
        region_names=("foreground", "boundary"),
        case_ids=np.asarray(["case_001"] * 60, dtype=object),
        epochs=epochs,
        coords=np.zeros((60, 2), dtype=np.int32),
        spatial_shapes=np.tile(np.asarray([[10, 10]], dtype=np.int32), (60, 1)),
    )

    sampled = downsample_feature_space(space, max_points=6, seed=7)

    assert sampled.features.shape[0] == 6
    assert sampled.spatial_shapes is not None
    assert sampled.spatial_shapes.shape == sampled.coords.shape
    strata = set(zip(sampled.epochs.tolist(), sampled.region_ids.tolist()))
    assert strata == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    }
