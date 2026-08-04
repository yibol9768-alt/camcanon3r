import numpy as np
import pytest

from camcanon3r.reliability import (
    binary_auroc,
    reliability_summary,
    resolve_case_field,
    risk_coverage_curve,
)


def test_resolve_case_field_supports_nested_and_flat_records() -> None:
    nested = {"scene": "office", "scores": {"rotation": 3.0}}
    flat = {"scores.rotation": 4.0}
    assert resolve_case_field(nested, "scores.rotation") == 3.0
    assert resolve_case_field(flat, "scores.rotation") == 4.0
    with pytest.raises(KeyError, match="missing at 'missing'"):
        resolve_case_field(nested, "missing.rotation")


def test_binary_auroc_is_exact_and_tie_aware() -> None:
    assert binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auroc([0, 1], [0.5, 0.5]) == 0.5
    assert binary_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_risk_coverage_groups_ties_without_order_dependence() -> None:
    first = risk_coverage_curve([1.0, 3.0, 10.0], [0.1, 0.1, 0.9])
    second = risk_coverage_curve([3.0, 1.0, 10.0], [0.1, 0.1, 0.9])
    assert first == second
    assert first["coverage"] == [pytest.approx(2 / 3), 1.0]
    assert first["risk"] == [2.0, pytest.approx(14 / 3)]
    assert first["aurc"] == pytest.approx(26 / 9)


def test_reliability_summary_bootstraps_scene_clusters() -> None:
    result = reliability_summary(
        errors=[0.1, 0.2, 3.0, 4.0, 0.3, 5.0],
        uncertainties=[0.1, 0.2, 0.8, 0.9, 0.3, 1.0],
        scenes=["a", "a", "b", "b", "c", "c"],
        failure_threshold=2.0,
        bootstrap_replicates=300,
        bootstrap_seed=9,
    )
    assert result["auroc"]["point_estimate"] == 1.0
    assert result["auroc"]["status"] == "defined"
    assert result["failure_count"] == 3
    assert result["bootstrap"]["resampling_unit"] == "scene_cluster"
    assert result["excess_aurc"]["point_estimate"] == pytest.approx(0.0)
    assert result["warnings"][0].startswith("descriptive_only")


def test_reliability_summary_preserves_undefined_single_class() -> None:
    result = reliability_summary(
        errors=[0.1, 0.2],
        uncertainties=[0.4, 0.5],
        scenes=["a", "b"],
        failure_threshold=2.0,
        bootstrap_replicates=20,
    )
    assert result["auroc"]["point_estimate"] is None
    assert result["auroc"]["status"] == "undefined_single_class"
    assert result["auroc"]["valid_replicates"] == 0
    assert result["auroc"]["undefined_replicates"] == 20


def test_reliability_metrics_validate_inputs() -> None:
    with pytest.raises(ValueError, match="only one class"):
        binary_auroc([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="non-negative"):
        risk_coverage_curve([-1.0], [0.1])
    with pytest.raises(ValueError, match="equal length"):
        reliability_summary(
            errors=[1.0],
            uncertainties=[0.1],
            scenes=["a", "b"],
            failure_threshold=0.5,
            bootstrap_replicates=2,
        )
    with pytest.raises(ValueError, match="non-finite"):
        reliability_summary(
            errors=[np.nan],
            uncertainties=[0.1],
            scenes=["a"],
            failure_threshold=0.5,
            bootstrap_replicates=2,
        )
