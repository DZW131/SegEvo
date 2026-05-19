import numpy as np

from segevo.sampling import REGION_NAMES, build_region_masks, sample_feature_regions


def test_build_region_masks_has_expected_regions():
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    pred = gt.copy()
    pred[1, 1] = 1
    pred[5, 5] = 0

    regions = build_region_masks(gt, pred, target_shape=(8, 8))

    assert set(regions) == set(REGION_NAMES)
    assert regions["foreground"].shape == (8, 8)
    assert regions["false_positive"].any()
    assert regions["false_negative"].any()
    assert regions["hard_background"].any()


def test_sample_feature_regions_returns_vectors_labels_and_coords():
    feature = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    pred = gt.copy()
    pred[1, 1] = 1
    pred[5, 5] = 0

    samples = sample_feature_regions(
        feature,
        gt,
        pred,
        max_samples_per_region=3,
        seed=7,
    )

    assert samples is not None
    assert samples.features.shape[1] == 2
    assert samples.coords.shape[1] == 2
    assert samples.features.shape[0] == samples.region_ids.shape[0] == samples.coords.shape[0]
    assert set(samples.region_ids.tolist()).issubset(set(range(len(REGION_NAMES))))
    for region_id in np.unique(samples.region_ids):
        assert np.count_nonzero(samples.region_ids == region_id) <= 3

