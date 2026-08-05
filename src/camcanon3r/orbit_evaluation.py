"""Camera-only ground-truth evaluation for canonical orbit projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .metrics import pairwise_relative_pose_errors
from .statistics import scene_bootstrap_summary

REQUIRED_METHODS = (
    "identity",
    "analytic_repair",
    "robust_projection",
    "uniform_projection",
    "orbit_medoid",
    "native_confidence",
    "oracle",
)


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    return {
        "count": len(finite),
        "median": float(np.median(finite)) if len(finite) else None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
    }


def evaluate_camera_extrinsics(
    target: ArrayLike,
    predicted: ArrayLike,
    *,
    translation_available: bool,
) -> dict[str, Any]:
    """Evaluate a camera-only output without attaching unrelated dense geometry."""

    errors = pairwise_relative_pose_errors(target, predicted)
    translation = (
        _summary(errors["translation_direction_degrees"])
        if translation_available
        else {
            "count": 0,
            "median": None,
            "mean": None,
            "p90": None,
            "status": "not_evaluated_projection_translation_unavailable",
        }
    )
    return {
        "pair_count": len(errors["pairs"]),
        "relative_rotation_degrees": _summary(errors["rotation_degrees"]),
        "translation_direction_degrees": translation,
        "camera_only": True,
    }


def select_ground_truth_oracle(
    target: ArrayLike,
    members: Mapping[str, ArrayLike],
    *,
    member_order: Sequence[str],
) -> tuple[str, dict[str, float]]:
    """Select the minimum GT rotation member as a reported upper bound only."""

    labels = [str(label) for label in member_order]
    if set(members) != set(labels):
        raise ValueError("oracle members do not match member order")
    errors = {
        label: float(
            evaluate_camera_extrinsics(
                target, members[label], translation_available=True
            )["relative_rotation_degrees"]["median"]
        )
        for label in labels
    }
    order_index = {label: index for index, label in enumerate(labels)}
    selected = min(labels, key=lambda label: (errors[label], order_index[label]))
    return selected, errors


def _ratio_bootstrap(
    denominator: np.ndarray,
    numerator: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    minimum_gap: float = 1e-12,
) -> dict[str, Any]:
    median_denominator = float(np.median(denominator))
    median_numerator = float(np.median(numerator))
    point = (
        median_numerator / median_denominator
        if median_denominator > minimum_gap
        else None
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(denominator), size=(replicates, len(denominator)))
    sampled_denominator = np.median(denominator[indices], axis=1)
    sampled_numerator = np.median(numerator[indices], axis=1)
    valid = sampled_denominator > minimum_gap
    ratios = sampled_numerator[valid] / sampled_denominator[valid]
    alpha = (1.0 - confidence_level) / 2.0
    lower = upper = None
    if len(ratios):
        lower, upper = (
            float(value) for value in np.quantile(ratios, [alpha, 1.0 - alpha])
        )
    return {
        "point_estimate": point,
        "lower": lower,
        "upper": upper,
        "valid_replicates": int(np.count_nonzero(valid)),
        "undefined_replicates": int(np.count_nonzero(~valid)),
    }


def summarize_orbit_camera_evaluations(
    per_scene: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    minimum_residual_gap_reduction: float,
    maximum_median_error_increase_degrees: float,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Aggregate scene-level rotation outcomes and apply the frozen promotion gate."""

    scenes = sorted(str(scene) for scene in per_scene)
    if not scenes or len(scenes) != len(per_scene):
        raise ValueError("orbit evaluation requires unique scene labels")
    errors: dict[str, np.ndarray] = {}
    for method in REQUIRED_METHODS:
        values = []
        for scene in scenes:
            record = per_scene[scene].get(method)
            if not isinstance(record, Mapping):
                raise TypeError(f"scene {scene} has no method {method}")
            value = record.get("relative_rotation_degrees", {}).get("median")
            if value is None or not np.isfinite(float(value)):
                raise ValueError(
                    f"scene {scene}/{method} has no finite rotation median"
                )
            values.append(float(value))
        errors[method] = np.asarray(values, dtype=np.float64)

    derived = {
        "robust_minus_analytic": errors["robust_projection"]
        - errors["analytic_repair"],
        "robust_minus_uniform": errors["robust_projection"]
        - errors["uniform_projection"],
        "robust_minus_medoid": errors["robust_projection"] - errors["orbit_medoid"],
        "analytic_residual_from_identity": errors["analytic_repair"]
        - errors["identity"],
        "robust_improvement_over_analytic": errors["analytic_repair"]
        - errors["robust_projection"],
    }
    raw_bootstrap = scene_bootstrap_summary(
        {**errors, **derived},
        scenes=scenes,
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        statistic="median",
    )
    recovery = _ratio_bootstrap(
        derived["analytic_residual_from_identity"],
        derived["robust_improvement_over_analytic"],
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
    )
    median_errors = {
        method: float(np.median(values)) for method, values in errors.items()
    }
    robust_minus_analytic = float(np.median(derived["robust_minus_analytic"]))
    robust_minus_uniform = float(np.median(derived["robust_minus_uniform"]))
    robust_minus_medoid = float(np.median(derived["robust_minus_medoid"]))
    gate = {
        "residual_gap_reduction_threshold": minimum_residual_gap_reduction,
        "residual_gap_reduction": recovery,
        "residual_gap_reduction_pass": recovery["point_estimate"] is not None
        and float(recovery["point_estimate"]) >= minimum_residual_gap_reduction,
        "maximum_median_error_increase_degrees": (
            maximum_median_error_increase_degrees
        ),
        "median_error_increase_degrees": robust_minus_analytic,
        "nondegradation_pass": (
            robust_minus_analytic <= maximum_median_error_increase_degrees
        ),
        "robust_minus_uniform_degrees": robust_minus_uniform,
        "beat_or_tie_uniform_pass": robust_minus_uniform <= 0.0,
        "robust_minus_medoid_degrees": robust_minus_medoid,
        "beat_or_tie_medoid_pass": robust_minus_medoid <= 0.0,
    }
    gate["promotion_pass"] = bool(
        gate["residual_gap_reduction_pass"]
        and gate["nondegradation_pass"]
        and gate["beat_or_tie_uniform_pass"]
        and gate["beat_or_tie_medoid_pass"]
    )
    return {
        "schema_version": "canonical-orbit-evaluation-summary-0.1",
        "scene_count": len(scenes),
        "scenes": scenes,
        "median_rotation_degrees": median_errors,
        "scene_cluster_bootstrap": raw_bootstrap,
        "promotion": gate,
    }
