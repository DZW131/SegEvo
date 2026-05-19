import numpy as np

from segevo.dashboard import _error_overlay, _overlay


def test_overlay_preserves_rgb_image_shape():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[..., 0] = 0.8
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    rendered = _overlay(image, mask, color=(52, 199, 89))

    assert rendered.shape == (8, 8, 3)
    assert rendered.dtype == np.float32
    assert rendered[3, 3, 1] > rendered[0, 0, 1]


def test_error_overlay_preserves_rgb_image_shape():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    error = np.zeros((8, 8), dtype=np.uint8)
    error[2:6, 2:6] = 3

    rendered = _error_overlay(image, error)

    assert rendered.shape == (8, 8, 3)
    assert rendered[3, 3, 0] > rendered[0, 0, 0]

