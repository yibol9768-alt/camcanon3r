import pytest

from camcanon3r.statistics import scene_bootstrap_summary


def test_scene_bootstrap_is_deterministic_and_paired() -> None:
    arguments = {
        "metrics": {"raw": [1.0, 3.0, 8.0], "delta": [0.5, 1.5, 4.0]},
        "scenes": ["a", "b", "c"],
        "replicates": 500,
        "seed": 23,
    }
    first = scene_bootstrap_summary(**arguments)
    second = scene_bootstrap_summary(**arguments)
    assert first == second
    assert first["metrics"]["raw"]["point_estimate"] == 3.0
    assert first["metrics"]["delta"]["point_estimate"] == 1.5
    assert first["metrics"]["raw"]["lower"] <= 3.0
    assert first["metrics"]["raw"]["upper"] >= 3.0


def test_scene_bootstrap_validates_cluster_contract() -> None:
    with pytest.raises(ValueError, match="unique"):
        scene_bootstrap_summary({"error": [1.0, 2.0]}, scenes=["a", "a"])
    with pytest.raises(ValueError, match="one value per scene"):
        scene_bootstrap_summary({"error": [1.0]}, scenes=["a", "b"])
    with pytest.raises(ValueError, match="non-finite"):
        scene_bootstrap_summary({"error": [float("nan")]}, scenes=["a"])
