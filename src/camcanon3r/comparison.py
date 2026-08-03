"""Gauge-invariant comparison of model-neutral prediction archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import aligned_depth_consistency, pairwise_relative_pose_errors


def _summarize(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0, "median": None, "mean": None, "p90": None}
    return {
        "count": len(finite),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def _finite_or_none(value: np.floating) -> float | None:
    return float(value) if np.isfinite(value) else None


def compare_predictions(
    reference_path: Path,
    candidate_path: Path,
    *,
    reference_label: str = "identity",
    candidate_label: str | None = None,
) -> dict[str, object]:
    """Compare predictions after cancelling world-frame and scale gauge."""

    with np.load(reference_path) as reference_data:
        reference = reference_data["extrinsic"]
        reference_depth = reference_data["depth"]
        reference_affine = reference_data["source_to_model_affine"]
    with np.load(candidate_path) as candidate_data:
        candidate = candidate_data["extrinsic"]
        candidate_depth = candidate_data["depth"]
        candidate_affine = candidate_data["source_to_model_affine"]

    errors = pairwise_relative_pose_errors(reference, candidate)
    depth = aligned_depth_consistency(
        reference_depth,
        candidate_depth,
        reference_affine,
        candidate_affine,
    )
    pairs = errors["pairs"]
    return {
        "reference": str(reference_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "reference_label": reference_label,
        "candidate_label": candidate_label or candidate_path.stem,
        "view_count": int(max(pairs.reshape(-1)) + 1),
        "pair_count": len(pairs),
        "rotation_degrees": _summarize(errors["rotation_degrees"]),
        "translation_direction_degrees": _summarize(
            errors["translation_direction_degrees"]
        ),
        "aligned_depth_consistency": depth,
        "per_pair": [
            {
                "views": [int(first), int(second)],
                "rotation_degrees": _finite_or_none(rotation),
                "translation_direction_degrees": _finite_or_none(translation),
            }
            for (first, second), rotation, translation in zip(
                pairs,
                errors["rotation_degrees"],
                errors["translation_direction_degrees"],
                strict=True,
            )
        ],
    }


def compare_vggt_predictions(
    reference_path: Path,
    candidate_path: Path,
    *,
    reference_label: str = "identity",
    candidate_label: str | None = None,
) -> dict[str, object]:
    """Backward-compatible alias for legacy scripts and stored notebooks."""

    return compare_predictions(
        reference_path,
        candidate_path,
        reference_label=reference_label,
        candidate_label=candidate_label,
    )
