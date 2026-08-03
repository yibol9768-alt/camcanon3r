"""Ground-truth gap-recovery accounting for analytic repairs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

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
}


def _nested_float(
    record: Mapping[str, Any], path: Sequence[str]
) -> float | None:
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
        None
        if clean_control_error is None
        else clean_control_error - identity_error
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
