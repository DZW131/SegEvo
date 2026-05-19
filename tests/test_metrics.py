import math

import numpy as np

from segevo.metrics import boundary_dice, dice_score, error_map, hd95, surface_dice, volume_error


def test_dice_empty_masks_are_perfect():
    empty = np.zeros((8, 8), dtype=np.uint8)
    assert dice_score(empty, empty) == 1.0


def test_error_map_labels_fp_and_fn():
    gt = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    pred = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    err = error_map(pred, gt)
    assert err.tolist() == [[1, 2], [3, 0]]


def test_volume_error_is_relative_to_gt():
    gt = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    pred = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)
    assert math.isclose(volume_error(pred, gt), 0.5)


def test_surface_metrics_for_identical_masks():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    assert hd95(mask, mask) == 0.0
    assert surface_dice(mask, mask) == 1.0
    assert boundary_dice(mask, mask) == 1.0


def test_boundary_dice_penalizes_shifted_boundary():
    gt = np.zeros((16, 16), dtype=np.uint8)
    pred = np.zeros((16, 16), dtype=np.uint8)
    gt[4:12, 4:12] = 1
    pred[5:13, 5:13] = 1
    assert 0.0 < boundary_dice(pred, gt) < 1.0
