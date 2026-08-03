"""Ground-truth reliability metrics with scene-cluster uncertainty."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def _finite_vector(values: ArrayLike, *, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


def binary_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Return exact AUROC with average ranks for tied scores.

    A larger score must indicate stronger belief in the positive class.
    """

    binary = np.asarray(labels)
    score = _finite_vector(scores, label="scores")
    if binary.shape != score.shape:
        raise ValueError("labels and scores must have the same shape")
    if not np.isin(binary, [0, 1, False, True]).all():
        raise ValueError("labels must be binary")
    positive = binary.astype(bool)
    positive_count = int(positive.sum())
    negative_count = len(positive) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC is undefined when only one class is present")

    _, inverse, counts = np.unique(
        score, return_inverse=True, return_counts=True
    )
    group_ends = np.cumsum(counts)
    group_starts = group_ends - counts + 1
    average_ranks = (group_starts + group_ends) / 2.0
    ranks = average_ranks[inverse]
    rank_sum = float(ranks[positive].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def risk_coverage_curve(
    errors: ArrayLike, uncertainties: ArrayLike
) -> dict[str, object]:
    """Retain low-uncertainty cases and aggregate complete tie blocks.

    AURC is a right-continuous step integral.  Grouping identical uncertainty
    scores makes the result invariant to arbitrary ordering within ties.
    """

    error = _finite_vector(errors, label="errors")
    uncertainty = _finite_vector(uncertainties, label="uncertainties")
    if error.shape != uncertainty.shape:
        raise ValueError("errors and uncertainties must have the same shape")
    if np.any(error < 0.0):
        raise ValueError("errors must be non-negative")

    order = np.argsort(uncertainty, kind="mergesort")
    ordered_error = error[order]
    ordered_uncertainty = uncertainty[order]
    cumulative_error = np.cumsum(ordered_error)
    group_ends = np.flatnonzero(
        np.r_[ordered_uncertainty[1:] != ordered_uncertainty[:-1], True]
    )
    retained_counts = group_ends + 1
    thresholds = ordered_uncertainty[group_ends]
    coverages = retained_counts / len(error)
    risks = cumulative_error[group_ends] / retained_counts
    coverage_widths = np.diff(np.r_[0.0, coverages])
    aurc = float(coverage_widths @ risks)
    return {
        "case_count": len(error),
        "thresholds": thresholds.tolist(),
        "coverage": coverages.tolist(),
        "risk": risks.tolist(),
        "aurc": aurc,
    }


def _percentile_interval(
    values: Sequence[float], confidence_level: float
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"lower": None, "upper": None, "valid_replicates": 0}
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(array, [alpha, 1.0 - alpha])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": len(array),
    }


def reliability_summary(
    errors: ArrayLike,
    uncertainties: ArrayLike,
    *,
    scenes: Sequence[str],
    failure_threshold: float,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 17,
) -> dict[str, object]:
    """Evaluate a failure score with a scene-cluster bootstrap."""

    error = _finite_vector(errors, label="errors")
    uncertainty = _finite_vector(uncertainties, label="uncertainties")
    scene = np.asarray([str(value) for value in scenes], dtype=object)
    if error.shape != uncertainty.shape or error.shape != scene.shape:
        raise ValueError("errors, uncertainties, and scenes must have equal length")
    if np.any(error < 0.0):
        raise ValueError("errors must be non-negative")
    if not np.isfinite(failure_threshold) or failure_threshold < 0.0:
        raise ValueError("failure_threshold must be finite and non-negative")
    if not isinstance(bootstrap_replicates, int) or bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")

    scene_labels = sorted(set(scene.tolist()))
    if not scene_labels:
        raise ValueError("at least one scene is required")
    cluster_indices = {
        label: np.flatnonzero(scene == label) for label in scene_labels
    }
    failures = error > failure_threshold
    curve = risk_coverage_curve(error, uncertainty)
    oracle_curve = risk_coverage_curve(error, error)
    point_auroc: float | None
    auroc_status: str
    try:
        point_auroc = binary_auroc(failures, uncertainty)
        auroc_status = "defined"
    except ValueError:
        point_auroc = None
        auroc_status = "undefined_single_class"

    rng = np.random.default_rng(bootstrap_seed)
    auroc_replicates: list[float] = []
    aurc_replicates: list[float] = []
    oracle_aurc_replicates: list[float] = []
    excess_aurc_replicates: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled_scenes = rng.integers(0, len(scene_labels), size=len(scene_labels))
        sampled_indices = np.concatenate(
            [cluster_indices[scene_labels[index]] for index in sampled_scenes]
        )
        sampled_error = error[sampled_indices]
        sampled_uncertainty = uncertainty[sampled_indices]
        sampled_failures = sampled_error > failure_threshold
        try:
            auroc_replicates.append(
                binary_auroc(sampled_failures, sampled_uncertainty)
            )
        except ValueError:
            pass
        sampled_curve = risk_coverage_curve(sampled_error, sampled_uncertainty)
        sampled_oracle = risk_coverage_curve(sampled_error, sampled_error)
        sampled_aurc = float(sampled_curve["aurc"])
        sampled_oracle_aurc = float(sampled_oracle["aurc"])
        aurc_replicates.append(sampled_aurc)
        oracle_aurc_replicates.append(sampled_oracle_aurc)
        excess_aurc_replicates.append(sampled_aurc - sampled_oracle_aurc)

    auroc_interval = _percentile_interval(auroc_replicates, confidence_level)
    aurc_interval = _percentile_interval(aurc_replicates, confidence_level)
    oracle_interval = _percentile_interval(
        oracle_aurc_replicates, confidence_level
    )
    excess_interval = _percentile_interval(
        excess_aurc_replicates, confidence_level
    )
    point_aurc = float(curve["aurc"])
    point_oracle_aurc = float(oracle_curve["aurc"])
    auroc_interval.update(
        {
            "point_estimate": point_auroc,
            "status": auroc_status,
            "undefined_replicates": (
                bootstrap_replicates - len(auroc_replicates)
            ),
        }
    )
    aurc_interval["point_estimate"] = point_aurc
    oracle_interval["point_estimate"] = point_oracle_aurc
    excess_interval["point_estimate"] = point_aurc - point_oracle_aurc

    warnings: list[str] = []
    if len(scene_labels) < 10:
        warnings.append("descriptive_only_fewer_than_10_scenes")
    if len(auroc_replicates) < 0.9 * bootstrap_replicates:
        warnings.append("more_than_10_percent_auroc_replicates_undefined")
    return {
        "score_direction": "higher_uncertainty_predicts_higher_error",
        "failure_definition": {
            "error_operator": ">",
            "threshold": failure_threshold,
        },
        "case_count": len(error),
        "scene_count": len(scene_labels),
        "scenes": scene_labels,
        "failure_count": int(failures.sum()),
        "failure_prevalence": float(failures.mean()),
        "auroc": auroc_interval,
        "risk_coverage": curve,
        "aurc": aurc_interval,
        "oracle_aurc": oracle_interval,
        "excess_aurc": excess_interval,
        "bootstrap": {
            "resampling_unit": "scene_cluster",
            "interval_method": "percentile",
            "confidence_level": confidence_level,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "warnings": warnings,
    }
