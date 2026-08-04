import pytest

from camcanon3r.repair_evaluation import (
    evaluate_repair_records,
    one_metric_gap_recovery,
    summarize_repair_evaluations,
)


def test_gap_recovery_reports_raw_values_and_unclipped_recovery() -> None:
    result = one_metric_gap_recovery(2.0, 6.0, 3.0, clean_control_error=2.02)
    assert result["corruption_gap"] == 4.0
    assert result["recovered_gap"] == 3.0
    assert result["gap_recovery"] == 0.75
    assert result["clean_delta"] == pytest.approx(0.02)
    assert result["clean_relative_degradation"] == pytest.approx(0.01)

    overshoot = one_metric_gap_recovery(2.0, 6.0, 1.0)
    assert overshoot["gap_recovery"] == 1.25


def test_gap_recovery_is_undefined_for_nonpositive_gap() -> None:
    result = one_metric_gap_recovery(2.0, 1.5, 1.0)
    assert result["gap_recovery"] is None
    assert result["corruption_gap"] == -0.5
    assert result["gap_recovery_status"].startswith("undefined")


def test_repair_records_preserve_pose_and_optional_depth_boundary() -> None:
    def record(rotation: float, translation: float, depth: float | None):
        return {
            "prediction": f"prediction-{rotation}",
            "relative_rotation_degrees": {"median": rotation},
            "translation_direction_degrees": {"median": translation},
            "depth": None if depth is None else {"mean_abs_rel": depth},
        }

    result = evaluate_repair_records(
        record(1.0, 2.0, None),
        record(5.0, 6.0, None),
        record(3.0, 4.0, None),
        clean_control=record(1.01, 2.02, None),
    )
    metrics = result["metrics"]
    assert metrics["relative_rotation_median_degrees"]["gap_recovery"] == 0.5
    assert metrics["translation_direction_median_degrees"]["gap_recovery"] == 0.5
    assert metrics["depth_mean_abs_rel"]["status"] == "unavailable"
    assert metrics["point_accuracy_mean_meters"]["status"] == "unavailable"


def test_repair_records_include_point_accuracy_and_completeness() -> None:
    def record(accuracy: float, completeness: float):
        return {
            "point_cloud": {
                "accuracy_meters": {"mean": accuracy},
                "completeness_meters": {"mean": completeness},
            }
        }

    result = evaluate_repair_records(
        record(0.10, 0.20),
        record(0.30, 0.50),
        record(0.15, 0.30),
    )
    assert result["metrics"]["point_accuracy_mean_meters"][
        "gap_recovery"
    ] == pytest.approx(0.75)
    assert result["metrics"]["point_completeness_mean_meters"][
        "gap_recovery"
    ] == pytest.approx(2.0 / 3.0)


def test_repair_records_include_dtu_point_units_without_conversion() -> None:
    def record(accuracy: float, completeness: float):
        return {
            "point_cloud": {
                "accuracy_millimeters": {"mean": accuracy},
                "completeness_millimeters": {"mean": completeness},
            }
        }

    result = evaluate_repair_records(
        record(1.0, 2.0),
        record(5.0, 8.0),
        record(3.0, 4.0),
    )
    assert result["metrics"]["point_accuracy_mean_millimeters"][
        "gap_recovery"
    ] == pytest.approx(0.5)
    assert result["metrics"]["point_completeness_mean_millimeters"][
        "gap_recovery"
    ] == pytest.approx(2.0 / 3.0)


def test_gap_recovery_rejects_negative_errors() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        one_metric_gap_recovery(0.0, -1.0, 0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        one_metric_gap_recovery(0.0, float("nan"), 0.0)


def test_repair_summary_bootstraps_paired_aggregate_recovery() -> None:
    def record(value: float):
        return {
            "prediction": f"prediction-{value}",
            "relative_rotation_degrees": {"median": value},
            "translation_direction_degrees": {"median": value * 2.0},
            "depth": {"mean_abs_rel": value / 100.0},
        }

    scenes = {
        "first": (record(1.0), record(5.0), record(3.0), record(1.01)),
        "second": (record(2.0), record(8.0), record(4.0), record(2.02)),
        "third": (record(3.0), record(7.0), record(4.0), record(3.03)),
    }
    summary = summarize_repair_evaluations(
        scenes, bootstrap_replicates=200, bootstrap_seed=17
    )
    rotation = summary["by_metric"]["relative_rotation_median_degrees"]
    assert rotation["scene_bootstrap"]["metrics"]["corruption_gap"][
        "point_estimate"
    ] == pytest.approx(4.0)
    assert rotation["scene_bootstrap"]["metrics"]["recovered_gap"][
        "point_estimate"
    ] == pytest.approx(3.0)
    assert rotation["gap_recovery"]["point_estimate"] == pytest.approx(0.75)
    assert rotation["promotion_gate"]["point_recovery_pass"] is True
    assert rotation["promotion_gate"]["point_clean_cost_pass"] is True
    assert summary["by_metric"]["point_accuracy_mean_meters"]["status"] == (
        "unavailable"
    )


def test_repair_summary_preserves_missing_clean_control_without_subset_bootstrap() -> None:
    identity = {
        "relative_rotation_degrees": {"median": 1.0},
        "translation_direction_degrees": {"median": 2.0},
        "depth": None,
    }
    corrupt = {
        "relative_rotation_degrees": {"median": 4.0},
        "translation_direction_degrees": {"median": 5.0},
        "depth": None,
    }
    repaired = {
        "relative_rotation_degrees": {"median": 2.0},
        "translation_direction_degrees": {"median": 3.0},
        "depth": None,
    }
    clean_missing = {
        "relative_rotation_degrees": {"median": None},
        "translation_direction_degrees": {"median": 2.0},
        "depth": None,
    }
    summary = summarize_repair_evaluations(
        {"scene": (identity, corrupt, repaired, clean_missing)},
        bootstrap_replicates=10,
    )
    rotation = summary["by_metric"]["relative_rotation_median_degrees"]
    assert rotation["status"] == "partially_unavailable"
    assert rotation["aggregation"] is None
    assert rotation["metric_availability"] == {
        "core_valid_scene_count": 1,
        "clean_control_valid_scene_count": 0,
        "complete_scene_count": 0,
        "undefined_scene_count": 1,
        "included_in_scene_bootstrap": False,
    }


def test_repair_summary_keeps_partial_secondary_metric_explicit() -> None:
    def record(rotation: float, point: float | None):
        return {
            "relative_rotation_degrees": {"median": rotation},
            "translation_direction_degrees": {"median": rotation * 2.0},
            "depth": {"mean_abs_rel": rotation / 100.0},
            "point_cloud": {
                "accuracy_meters": {"mean": point},
                "completeness_meters": {"mean": point},
            },
        }

    scenes = {
        "first": (
            record(1.0, 0.1),
            record(5.0, 0.5),
            record(2.0, 0.2),
            record(1.01, 0.101),
        ),
        "second": (
            record(1.0, 0.1),
            record(5.0, 0.5),
            record(2.0, None),
            record(1.01, 0.101),
        ),
    }
    summary = summarize_repair_evaluations(
        scenes, bootstrap_replicates=20
    )
    point = summary["by_metric"]["point_accuracy_mean_meters"]
    assert point["status"] == "partially_unavailable"
    assert point["scenes"] == ["first"]
    assert point["metric_availability"] == {
        "core_valid_scene_count": 1,
        "clean_control_valid_scene_count": 1,
        "complete_scene_count": 1,
        "undefined_scene_count": 1,
        "included_in_scene_bootstrap": False,
    }
    assert summary["by_metric"]["relative_rotation_median_degrees"][
        "status"
    ] == "available"
