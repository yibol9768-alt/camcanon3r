import pytest

from camcanon3r.repair_evaluation import (
    evaluate_repair_records,
    one_metric_gap_recovery,
)


def test_gap_recovery_reports_raw_values_and_unclipped_recovery() -> None:
    result = one_metric_gap_recovery(
        2.0, 6.0, 3.0, clean_control_error=2.02
    )
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


def test_gap_recovery_rejects_negative_errors() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        one_metric_gap_recovery(0.0, -1.0, 0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        one_metric_gap_recovery(0.0, float("nan"), 0.0)
