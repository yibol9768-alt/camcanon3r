"""Aggregate scene-level CamCanon3R comparison records."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .statistics import scene_bootstrap_summary


def summarize_comparison_files(
    paths: list[Path],
    *,
    rotation_threshold: float = 2.0,
    depth_threshold: float = 0.05,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 17,
) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one comparison record is required")
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        reference = Path(record["reference"])
        row = {
            "scene": reference.parent.name,
            "candidate": record["candidate_label"],
            "rotation_median_degrees": record["rotation_degrees"]["median"],
            "translation_median_degrees": record[
                "translation_direction_degrees"
            ]["median"],
            "depth_mean_abs_rel": record["aligned_depth_consistency"][
                "mean_abs_rel"
            ],
            "valid_depth_pixels": record["aligned_depth_consistency"][
                "valid_pixels"
            ],
            "source": str(path),
        }
        rows.append(row)
        grouped[str(row["candidate"])].append(row)

    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        pair = (str(row["scene"]), str(row["candidate"]))
        if pair in seen_pairs:
            raise ValueError(
                "duplicate scene/candidate comparison record: "
                f"scene={pair[0]!r}, candidate={pair[1]!r}"
            )
        seen_pairs.add(pair)

    by_variant: dict[str, dict[str, object]] = {}
    for candidate, candidate_rows in sorted(grouped.items()):
        candidate_rows = sorted(candidate_rows, key=lambda row: str(row["scene"]))
        metric_arrays = {
            "rotation_median_degrees": np.asarray(
                [row["rotation_median_degrees"] for row in candidate_rows],
                dtype=np.float64,
            ),
            "translation_median_degrees": np.asarray(
                [row["translation_median_degrees"] for row in candidate_rows],
                dtype=np.float64,
            ),
            "depth_mean_abs_rel": np.asarray(
                [row["depth_mean_abs_rel"] for row in candidate_rows],
                dtype=np.float64,
            ),
        }
        complete_metrics = {
            label: values
            for label, values in metric_arrays.items()
            if np.isfinite(values).all()
        }
        metric_availability = {
            label: {
                "valid_scene_count": int(np.count_nonzero(np.isfinite(values))),
                "undefined_scene_count": int(
                    np.count_nonzero(~np.isfinite(values))
                ),
                "included_in_scene_bootstrap": label in complete_metrics,
            }
            for label, values in metric_arrays.items()
        }
        rotations = metric_arrays["rotation_median_degrees"]
        translations = metric_arrays["translation_median_degrees"]
        depths = metric_arrays["depth_mean_abs_rel"]
        by_variant[candidate] = {
            "scene_count": len(candidate_rows),
            "scenes": sorted(str(row["scene"]) for row in candidate_rows),
            "median_of_scene_rotation_medians_degrees": _complete_median(
                rotations
            ),
            "median_of_scene_translation_medians_degrees": _complete_median(
                translations
            ),
            "median_of_scene_depth_mean_abs_rel": _complete_median(depths),
            "scenes_over_rotation_threshold": int(
                np.count_nonzero(rotations > rotation_threshold)
            ),
            "scenes_over_depth_threshold": int(
                np.count_nonzero(depths > depth_threshold)
            ),
            "metric_availability": metric_availability,
            "scene_bootstrap": scene_bootstrap_summary(
                complete_metrics,
                scenes=[str(row["scene"]) for row in candidate_rows],
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            ),
        }
    return {
        "rotation_threshold_degrees": rotation_threshold,
        "depth_abs_rel_threshold": depth_threshold,
        "comparison_count": len(rows),
        "comparisons": rows,
        "by_variant": by_variant,
    }


def _complete_median(values: np.ndarray) -> float | None:
    if not np.isfinite(values).all():
        return None
    return float(np.median(values))


def summarize_eth3d_evaluations(
    paths: list[Path],
    *,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 17,
) -> dict[str, object]:
    """Aggregate paired multi-scene GT metrics and identity deltas."""

    if not paths:
        raise ValueError("at least one ETH3D evaluation record is required")
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        depth = record["depth"]
        rows.append(
            {
                "scene": str(record.get("scene", path.parent.name)),
                "variant": record.get("variant", Path(record["prediction"]).stem),
                "rotation_median_degrees": record["relative_rotation_degrees"][
                    "median"
                ],
                "translation_median_degrees": record[
                    "translation_direction_degrees"
                ]["median"],
                "depth_mean_abs_rel": depth["mean_abs_rel"] if depth else None,
                "valid_depth_pixels": depth["valid_pixels"] if depth else None,
                "source": str(path),
            }
        )

    by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        pair = (str(row["scene"]), str(row["variant"]))
        if pair in seen_pairs:
            raise ValueError(
                "duplicate ETH3D scene/variant evaluation: "
                f"scene={pair[0]!r}, variant={pair[1]!r}"
            )
        seen_pairs.add(pair)
        by_scene[pair[0]].append(row)

    identities: dict[str, dict[str, object]] = {}
    expected_variants: set[str] | None = None
    for scene, scene_rows in sorted(by_scene.items()):
        scene_variants = {str(row["variant"]) for row in scene_rows}
        if expected_variants is None:
            expected_variants = scene_variants
        elif scene_variants != expected_variants:
            missing = sorted(expected_variants - scene_variants)
            extra = sorted(scene_variants - expected_variants)
            raise ValueError(
                f"incomplete paired ETH3D design for scene {scene!r}: "
                f"missing={missing}, extra={extra}"
            )
        scene_identities = [row for row in scene_rows if row["variant"] == "identity"]
        if len(scene_identities) != 1:
            raise ValueError(
                f"ETH3D scene {scene!r} requires exactly one identity record"
            )
        identity = scene_identities[0]
        identities[scene] = identity
        identity_has_depth = identity["depth_mean_abs_rel"] is not None
        for row in scene_rows:
            row_has_depth = row["depth_mean_abs_rel"] is not None
            if row_has_depth != identity_has_depth:
                raise ValueError(
                    f"inconsistent depth availability in ETH3D scene {scene!r}"
                )
            row["rotation_delta_from_identity_degrees"] = float(
                row["rotation_median_degrees"]
                - identity["rotation_median_degrees"]
            )
            row["translation_delta_from_identity_degrees"] = float(
                row["translation_median_degrees"]
                - identity["translation_median_degrees"]
            )
            if row["depth_mean_abs_rel"] is None:
                row["depth_abs_rel_delta_from_identity"] = None
            else:
                row["depth_abs_rel_delta_from_identity"] = float(
                    row["depth_mean_abs_rel"]
                    - identity["depth_mean_abs_rel"]
                )

    depth_modes = {
        identity["depth_mean_abs_rel"] is not None for identity in identities.values()
    }
    if len(depth_modes) != 1:
        raise ValueError(
            "an ETH3D summary cannot mix pose-only and pose-plus-depth scenes"
        )
    depth_evaluated = next(iter(depth_modes))

    grouped_variants: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_variants[str(row["variant"])].append(row)
    by_variant: dict[str, dict[str, object]] = {}
    for variant, variant_rows in sorted(grouped_variants.items()):
        variant_rows = sorted(variant_rows, key=lambda row: str(row["scene"]))
        scenes = [str(row["scene"]) for row in variant_rows]
        metrics: dict[str, list[float]] = {
            "rotation_median_degrees": [
                float(row["rotation_median_degrees"]) for row in variant_rows
            ],
            "translation_median_degrees": [
                float(row["translation_median_degrees"]) for row in variant_rows
            ],
            "rotation_delta_from_identity_degrees": [
                float(row["rotation_delta_from_identity_degrees"])
                for row in variant_rows
            ],
            "translation_delta_from_identity_degrees": [
                float(row["translation_delta_from_identity_degrees"])
                for row in variant_rows
            ],
        }
        if depth_evaluated:
            metrics["depth_mean_abs_rel"] = [
                float(row["depth_mean_abs_rel"]) for row in variant_rows
            ]
            metrics["depth_abs_rel_delta_from_identity"] = [
                float(row["depth_abs_rel_delta_from_identity"])
                for row in variant_rows
            ]
        by_variant[variant] = {
            "scene_count": len(scenes),
            "scenes": scenes,
            "scene_bootstrap": scene_bootstrap_summary(
                metrics,
                scenes=scenes,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            ),
        }
    return {
        "evaluation_count": len(rows),
        "scene_count": len(by_scene),
        "scenes": sorted(by_scene),
        "depth_evaluated": depth_evaluated,
        "identity": next(iter(identities.values())) if len(identities) == 1 else None,
        "identities": identities,
        "evaluations": rows,
        "by_variant": by_variant,
    }
