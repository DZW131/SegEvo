from segevo.demo import generate_demo_run
from segevo.feature_space import available_feature_layers, load_feature_space


def test_demo_run_includes_feature_samples(tmp_path):
    run_dir = generate_demo_run(tmp_path / "demo", epochs=2, cases=1)

    assert available_feature_layers(run_dir) == ["bottleneck"]

    space = load_feature_space(run_dir, layer="bottleneck")
    assert space.features.shape[0] > 0
    assert space.features.shape[1] == 8

