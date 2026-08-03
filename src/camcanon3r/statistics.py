"""Deterministic scene-level uncertainty estimates for CamCanon3R."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def scene_bootstrap_summary(
    metrics: Mapping[str, Sequence[float]],
    *,
    scenes: Sequence[str],
    replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 17,
    statistic: str = "median",
) -> dict[str, object]:
    """Bootstrap a scene-level statistic with shared paired resamples.

    Every metric must contain exactly one finite value per scene in the same
    order.  One index matrix is shared across metrics so paired quantities such
    as raw error and identity-relative delta are resampled consistently.
    """

    scene_labels = [str(scene) for scene in scenes]
    if not scene_labels:
        raise ValueError("at least one scene is required")
    if len(set(scene_labels)) != len(scene_labels):
        raise ValueError("scene labels must be unique")
    if not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    if not metrics:
        raise ValueError("at least one metric is required")

    arrays: dict[str, np.ndarray] = {}
    for label, values in metrics.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(scene_labels),):
            raise ValueError(
                f"metric {label!r} must contain one value per scene; "
                f"expected {len(scene_labels)}, got shape {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"metric {label!r} contains a non-finite value")
        arrays[str(label)] = array

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(scene_labels),
        size=(replicates, len(scene_labels)),
    )
    reducer = np.median if statistic == "median" else np.mean
    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, dict[str, float]] = {}
    for label, values in arrays.items():
        bootstrap_values = reducer(values[indices], axis=1)
        lower, upper = np.quantile(bootstrap_values, [alpha, 1.0 - alpha])
        intervals[label] = {
            "point_estimate": float(reducer(values)),
            "lower": float(lower),
            "upper": float(upper),
        }

    return {
        "resampling_unit": "scene",
        "statistic": statistic,
        "interval_method": "percentile",
        "confidence_level": confidence_level,
        "replicates": replicates,
        "seed": seed,
        "scene_count": len(scene_labels),
        "scenes": scene_labels,
        "small_sample_warning": (
            "descriptive_only_fewer_than_10_scenes"
            if len(scene_labels) < 10
            else None
        ),
        "metrics": intervals,
    }
