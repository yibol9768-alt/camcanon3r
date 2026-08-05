import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from camcanon3r.orbit_evaluation import (
    evaluate_camera_extrinsics,
    select_ground_truth_oracle,
    summarize_orbit_camera_evaluations,
)


def _two_view(angle_degrees: float) -> np.ndarray:
    rotations = np.stack(
        [
            np.eye(3),
            Rotation.from_euler("z", angle_degrees, degrees=True).as_matrix(),
        ]
    )
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1]])
    translations = -np.einsum("vij,vj->vi", rotations, centers)
    return np.concatenate([rotations, translations[:, :, None]], axis=2)


def _record(error: float) -> dict[str, object]:
    return {
        "relative_rotation_degrees": {
            "count": 3,
            "median": error,
            "mean": error,
            "p90": error,
        }
    }


def test_camera_only_evaluation_respects_translation_boundary():
    target = _two_view(0.0)
    prediction = _two_view(5.0)

    record = evaluate_camera_extrinsics(target, prediction, translation_available=False)

    assert record["relative_rotation_degrees"]["median"] == pytest.approx(5.0)
    assert record["translation_direction_degrees"]["median"] is None
    assert record["translation_direction_degrees"]["status"].startswith("not_evaluated")


def test_ground_truth_oracle_uses_frozen_tie_order():
    target = _two_view(0.0)
    members = {
        "first": _two_view(2.0),
        "second": _two_view(-2.0),
        "third": _two_view(4.0),
    }

    selected, errors = select_ground_truth_oracle(
        target, members, member_order=("first", "second", "third")
    )

    assert selected == "first"
    assert errors == pytest.approx({"first": 2.0, "second": 2.0, "third": 4.0})


def test_summary_passes_a_real_residual_reduction():
    per_scene = {
        f"scene{index}": {
            "identity": _record(1.0),
            "analytic_repair": _record(3.0),
            "response_projection": _record(2.0),
            "robust_projection": _record(2.2),
            "uniform_projection": _record(2.5),
            "orbit_medoid": _record(2.4),
            "native_confidence": _record(2.8),
            "oracle": _record(1.5),
        }
        for index in range(10)
    }

    summary = summarize_orbit_camera_evaluations(
        per_scene,
        minimum_residual_gap_reduction=0.15,
        maximum_median_error_increase_degrees=0.1,
        bootstrap_replicates=100,
        confidence_level=0.95,
        bootstrap_seed=1701,
    )

    assert summary["median_rotation_degrees"]["response_projection"] == 2.0
    assert summary["promotion"]["residual_gap_reduction"]["point_estimate"] == 0.5
    assert summary["promotion"]["promotion_pass"] is True


def test_summary_retains_a_failed_projection_gate():
    per_scene = {
        f"scene{index}": {
            "identity": _record(1.0),
            "analytic_repair": _record(3.0),
            "response_projection": _record(3.3),
            "robust_projection": _record(3.0),
            "uniform_projection": _record(2.5),
            "orbit_medoid": _record(2.4),
            "native_confidence": _record(2.8),
            "oracle": _record(1.5),
        }
        for index in range(10)
    }

    summary = summarize_orbit_camera_evaluations(
        per_scene,
        minimum_residual_gap_reduction=0.15,
        maximum_median_error_increase_degrees=0.1,
        bootstrap_replicates=100,
        confidence_level=0.95,
        bootstrap_seed=1701,
    )

    assert summary["promotion"]["residual_gap_reduction_pass"] is False
    assert summary["promotion"]["nondegradation_pass"] is False
    assert summary["promotion"]["beat_or_tie_robust_group_pass"] is False
    assert summary["promotion"]["beat_or_tie_uniform_pass"] is False
    assert summary["promotion"]["promotion_pass"] is False
