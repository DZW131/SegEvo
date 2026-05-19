import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


def load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "pytorch_unet_training.py"
    spec = importlib.util.spec_from_file_location("pytorch_unet_training", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tiny_unet_forward_shape():
    module = load_example_module()
    model = module.TinyUNet(base_channels=4)
    sample = module.SyntheticLesionDataset(count=1, image_size=32)[0]
    output = model(sample.image.unsqueeze(0))
    assert output.shape == (1, 1, 32, 32)
