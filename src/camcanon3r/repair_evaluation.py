"""Ground-truth gap-recovery accounting for analytic repairs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_ERROR_PATHS = {
    "relative_rotation_median_degrees": (
        "relative_rotation_degrees",
        "median",
    ),
    "translation_direction_median_degrees": (
        "translation_direction_degrees",
        "median",
    ),
    "depth_mean_abs_rel": ("depth", "mean_abs_rel"),
    "point_accuracy_mean_meters": (
        "point_cloud",
        "accuracy_meters",
        "mean",
    ),
    "point_completeness_mean_meters": (
        "point_cloud",
        "completeness_meters",
        "mean",
    ),
    "point_accuracy_mean_millimeters": (
        "point_cloud",
        "accuracy_millimeters",
        "mean",
    ),
    "point_completeness_mean_millimeters": (
        "point_cloud",
        "completeness_millimeters",
        "mean",
    ),
}


def _nested_float(record: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = record
    for key in path:
        if value is None:
            return None
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            f"error metric at {'.'.join(path)} must be finite and non-negative"
        )
    return result


def one_metric_gap_recovery(
    identity_error: float,
    corrupt_error: float,
    repaired_error: float,
    *,
    clean_control_error: float | None = None,
    minimum_gap: float = 1e-12,
) -> dict[str, float | None | str]:
    """Report unmodified errors and recovery without clipping favorable values."""

    values = [identity_error, corrupt_error, repaired_error]
    if clean_control_error is not None:
        values.append(clean_control_error)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("error values must be finite and non-negative")
    if minimum_gap < 0:
        raise ValueError("minimum gap must be non-negative")
    gap = corrupt_error - identity_error
    recovered = corrupt_error - repaired_error
    if gap <= minimum_gap:
        recovery = None
        recovery_status = "undefined_nonpositive_or_noise_floor_gap"
    else:
        recovery = recovered / gap
        recovery_status = "defined"
    clean_delta = (
        None if clean_control_error is None else clean_control_error - identity_error
    )
    clean_relative = None
    if clean_control_error is not None and identity_error > minimum_gap:
        clean_relative = clean_delta / identity_error
    return {
        "identity_error": identity_error,
        "corrupt_error": corrupt_error,
        "repaired_error": repaired_error,
        "clean_control_error": clean_control_error,
        "corruption_gap": gap,
        "recovered_gap": recovered,
        "gap_recovery": recovery,
        "gap_recovery_status": recovery_status,
        "clean_delta": clean_delta,
        "clean_relative_degradation": clean_relative,
    }


def evaluate_repair_records(
    identity: Mapping[str, Any],
    corrupt: Mapping[str, Any],
    repaired: Mapping[str, Any],
    *,
    clean_control: Mapping[str, Any] | None = None,
    error_paths: Mapping[str, Sequence[str]] = DEFAULT_ERROR_PATHS,
    minimum_gap: float = 1e-12,
) -> dict[str, object]:
    """Compare matched GT evaluation records for one scene and intervention."""

    metrics: dict[str, object] = {}
    for label, path in error_paths.items():
        identity_error = _nested_float(identity, path)
        corrupt_error = _nested_float(corrupt, path)
        repaired_error = _nested_float(repaired, path)
        clean_error = (
            None if clean_control is None else _nested_float(clean_control, path)
        )
        required = [identity_error, corrupt_error, repaired_error]
        if any(value is None for value in required):
            metrics[label] = {
                "status": "unavailable",
                "path": list(path),
            }
            continue
        metric = one_metric_gap_recovery(
            float(identity_error),
            float(corrupt_error),
            float(repaired_error),
            clean_control_error=clean_error,
            minimum_gap=minimum_gap,
        )
        metric["status"] = "available"
        metric["path"] = list(path)
        metrics[label] = metric
    return {
        "identity_prediction": identity.get("prediction"),
        "corrupt_prediction": corrupt.get("prediction"),
        "repaired_prediction": repaired.get("prediction"),
        "clean_control_prediction": (
            None if clean_control is None else clean_control.get("prediction")
        ),
        "minimum_gap": minimum_gap,
        "metrics": metrics,
    }


def _ratio_bootstrap(
    corruption_gap: np.ndarray,
    recovered_gap: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    minimum_gap: float,
) -> dict[str, object]:
    if not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    median_gap = float(np.median(corruption_gap))
    median_recovered = float(np.median(recovered_gap))
    point = median_recovered / median_gap if median_gap > minimum_gap else None
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(corruption_gap), size=(replicates, len(corruption_gap))
    )
    sampled_gap = np.median(corruption_gap[indices], axis=1)
    sampled_recovered = np.median(recovered_gap[indices], axis=1)
    valid = sampled_gap > minimum_gap
    ratios = sampled_recovered[valid] / sampled_gap[valid]
    alpha = (1.0 - confidence_level) / 2.0
    lower: float | None = None
    upper: float | None = None
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


def summarize_repair_evaluations(
    scene_records: Mapping[
        str,
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ],
    ],
    *,
    minimum_gap: float = 1e-12,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 17,
    recovery_threshold: float = 0.30,
    clean_relative_threshold: float = 0.02,
) -> dict[str, object]:
    """Aggregate paired analytic-repair errors with scene-level bootstrap."""

    if not scene_records:
        raise ValueError("at least one repair scene is required")
    if recovery_threshold < 0.0 or clean_relative_threshold < 0.0:
        raise ValueError("repair thresholds must be non-negative")
    scenes = sorted(str(scene) for scene in scene_records)
    if len(scenes) != len(scene_records):
        raise ValueError("repair scene labels must be unique strings")
    per_scene: dict[str, dict[str, object]] = {}
    for scene in scenes:
        identity, corrupt, repaired, clean = scene_records[scene]
        per_scene[scene] = evaluate_repair_records(
            identity,
            corrupt,
            repaired,
            clean_control=clean,
            minimum_gap=minimum_gap,
        )

    metric_labels = list(DEFAULT_ERROR_PATHS)
    by_metric: dict[str, dict[str, object]] = {}
    for label in metric_labels:
        scene_metrics = [per_scene[scene]["metrics"][label] for scene in scenes]
        core_available = [
            record["status"] == "available" for record in scene_metrics
        ]
        clean_available = [
            bool(core) and record["clean_control_error"] is not None
            for core, record in zip(core_available, scene_metrics, strict=True)
        ]
        complete = [
            bool(core and clean)
            for core, clean in zip(core_available, clean_available, strict=True)
        ]
        metric_availability = {
            "core_valid_scene_count": int(sum(core_available)),
            "clean_control_valid_scene_count": int(sum(clean_available)),
            "complete_scene_count": int(sum(complete)),
            "undefined_scene_count": int(len(scenes) - sum(complete)),
            "included_in_scene_bootstrap": bool(all(complete)),
        }
        if not all(complete):
            complete_scenes = [
                scene
                for scene, is_complete in zip(scenes, complete, strict=True)
                if is_complete
            ]
            by_metric[label] = {
                "status": (
                    "partially_unavailable"
                    if any(core_available)
                    else "unavailable"
                ),
                "scene_count": len(complete_scenes),
                "scenes": complete_scenes,
                "metric_availability": metric_availability,
                "aggregation": None,
                "reason": (
                    "complete paired scene design required; no subset bootstrap"
                ),
            }
            continue
        fields = (
            "identity_error",
            "corrupt_error",
            "repaired_error",
            "clean_control_error",
            "corruption_gap",
            "recovered_gap",
            "clean_delta",
        )
        values = {
            field: np.asarray(
                [float(record[field]) for record in scene_metrics],
                dtype=np.float64,
            )
            for field in fields
        }
        if not all(np.isfinite(array).all() for array in values.values()):
            raise ValueError(f"repair metric contains a non-finite value: {label}")
        identity = values["identity_error"]
        relative_clean = np.full(len(identity), np.nan, dtype=np.float64)
        valid_identity = identity > minimum_gap
        relative_clean[valid_identity] = (
            values["clean_delta"][valid_identity] / identity[valid_identity]
        )
        complete_values = dict(values)
        if np.isfinite(relative_clean).all():
            complete_values["clean_relative_degradation"] = relative_clean

        rng = np.random.default_rng(bootstrap_seed)
        indices = rng.integers(0, len(scenes), size=(bootstrap_replicates, len(scenes)))
        alpha = (1.0 - confidence_level) / 2.0
        intervals: dict[str, dict[str, float]] = {}
        for field, array in complete_values.items():
            sampled = np.median(array[indices], axis=1)
            lower, upper = np.quantile(sampled, [alpha, 1.0 - alpha])
            intervals[field] = {
                "point_estimate": float(np.median(array)),
                "lower": float(lower),
                "upper": float(upper),
            }
        recovery = _ratio_bootstrap(
            values["corruption_gap"],
            values["recovered_gap"],
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
            minimum_gap=minimum_gap,
        )
        clean_relative = intervals.get("clean_relative_degradation")
        gate = {
            "recovery_threshold": recovery_threshold,
            "clean_relative_threshold": clean_relative_threshold,
            "point_recovery_pass": (
                recovery["point_estimate"] is not None
                and float(recovery["point_estimate"]) >= recovery_threshold
            ),
            "point_clean_cost_pass": (
                clean_relative is not None
                and clean_relative["point_estimate"] <= clean_relative_threshold
            ),
            "confidence_bound_recovery_pass": (
                recovery["lower"] is not None
                and float(recovery["lower"]) >= recovery_threshold
            ),
            "confidence_bound_clean_cost_pass": (
                clean_relative is not None
                and clean_relative["upper"] <= clean_relative_threshold
            ),
        }
        by_metric[label] = {
            "status": "available",
            "scene_count": len(scenes),
            "scenes": scenes,
            "metric_availability": metric_availability,
            "scene_bootstrap": {
                "resampling_unit": "scene",
                "statistic": "median",
                "interval_method": "percentile",
                "confidence_level": confidence_level,
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "small_sample_warning": (
                    "descriptive_only_fewer_than_10_scenes"
                    if len(scenes) < 10
                    else None
                ),
                "metrics": intervals,
            },
            "gap_recovery": recovery,
            "promotion_gate": gate,
        }
    return {
        "schema_version": "1.0",
        "scene_count": len(scenes),
        "scenes": scenes,
        "minimum_gap": minimum_gap,
        "recovery_definition": (
            "median_scene_recovered_gap / median_scene_corruption_gap"
        ),
        "recovery_threshold": recovery_threshold,
        "clean_relative_threshold": clean_relative_threshold,
        "per_scene": per_scene,
        "by_metric": by_metric,
    }
