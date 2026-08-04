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
            "translation_median_degrees": record["translation_direction_degrees"][
                "median"
            ],
            "depth_mean_abs_rel": record["aligned_depth_consistency"]["mean_abs_rel"],
            "valid_depth_pixels": record["aligned_depth_consistency"]["valid_pixels"],
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
                "undefined_scene_count": int(np.count_nonzero(~np.isfinite(values))),
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
            "median_of_scene_rotation_medians_degrees": _complete_median(rotations),
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
        point_cloud = record.get("point_cloud")
        if point_cloud is not None and depth is None:
            raise ValueError("ETH3D point-cloud metrics require raw depth evaluation")
        intrinsics = record["intrinsics"]
        rows.append(
            {
                "scene": str(record.get("scene", path.parent.name)),
                "variant": record.get("variant", Path(record["prediction"]).stem),
                "rotation_median_degrees": record["relative_rotation_degrees"][
                    "median"
                ],
                "translation_median_degrees": record["translation_direction_degrees"][
                    "median"
                ],
                "focal_relative_error_median": intrinsics["focal_relative_error"][
                    "median"
                ],
                "principal_point_normalized_error_median": intrinsics[
                    "principal_point_normalized_error"
                ]["median"],
                "depth_mean_abs_rel": depth["mean_abs_rel"] if depth else None,
                "valid_depth_pixels": depth["valid_pixels"] if depth else None,
                "point_accuracy_mean_meters": (
                    point_cloud["accuracy_meters"]["mean"]
                    if point_cloud is not None
                    else None
                ),
                "point_completeness_mean_meters": (
                    point_cloud["completeness_meters"]["mean"]
                    if point_cloud is not None
                    else None
                ),
                "point_cloud_evaluated": point_cloud is not None,
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
        identity_has_point_protocol = bool(identity["point_cloud_evaluated"])
        for row in scene_rows:
            row_has_depth = row["depth_mean_abs_rel"] is not None
            if row_has_depth != identity_has_depth:
                raise ValueError(
                    f"inconsistent depth availability in ETH3D scene {scene!r}"
                )
            row_has_point_protocol = bool(row["point_cloud_evaluated"])
            if row_has_point_protocol != identity_has_point_protocol:
                raise ValueError(
                    f"inconsistent point-cloud protocol in ETH3D scene {scene!r}"
                )
            row["rotation_delta_from_identity_degrees"] = _optional_difference(
                row["rotation_median_degrees"],
                identity["rotation_median_degrees"],
            )
            row["translation_delta_from_identity_degrees"] = _optional_difference(
                row["translation_median_degrees"],
                identity["translation_median_degrees"],
            )
            row["focal_relative_error_delta_from_identity"] = _optional_difference(
                row["focal_relative_error_median"],
                identity["focal_relative_error_median"],
            )
            row["principal_point_normalized_error_delta_from_identity"] = (
                _optional_difference(
                    row["principal_point_normalized_error_median"],
                    identity["principal_point_normalized_error_median"],
                )
            )
            if row["depth_mean_abs_rel"] is None:
                row["depth_abs_rel_delta_from_identity"] = None
            else:
                row["depth_abs_rel_delta_from_identity"] = float(
                    row["depth_mean_abs_rel"] - identity["depth_mean_abs_rel"]
                )
            row["point_accuracy_delta_from_identity_meters"] = _optional_difference(
                row["point_accuracy_mean_meters"],
                identity["point_accuracy_mean_meters"],
            )
            row["point_completeness_delta_from_identity_meters"] = _optional_difference(
                row["point_completeness_mean_meters"],
                identity["point_completeness_mean_meters"],
            )

    depth_modes = {
        identity["depth_mean_abs_rel"] is not None for identity in identities.values()
    }
    if len(depth_modes) != 1:
        raise ValueError(
            "an ETH3D summary cannot mix pose-only and pose-plus-depth scenes"
        )
    depth_evaluated = next(iter(depth_modes))
    point_modes = {
        bool(identity["point_cloud_evaluated"]) for identity in identities.values()
    }
    if len(point_modes) != 1:
        raise ValueError(
            "an ETH3D summary cannot mix scenes with and without point-cloud metrics"
        )
    point_cloud_evaluated = next(iter(point_modes))

    grouped_variants: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_variants[str(row["variant"])].append(row)
    by_variant: dict[str, dict[str, object]] = {}
    for variant, variant_rows in sorted(grouped_variants.items()):
        variant_rows = sorted(variant_rows, key=lambda row: str(row["scene"]))
        scenes = [str(row["scene"]) for row in variant_rows]
        metric_values: dict[str, list[object]] = {
            "rotation_median_degrees": [
                row["rotation_median_degrees"] for row in variant_rows
            ],
            "translation_median_degrees": [
                row["translation_median_degrees"] for row in variant_rows
            ],
            "focal_relative_error_median": [
                row["focal_relative_error_median"] for row in variant_rows
            ],
            "principal_point_normalized_error_median": [
                row["principal_point_normalized_error_median"] for row in variant_rows
            ],
            "rotation_delta_from_identity_degrees": [
                row["rotation_delta_from_identity_degrees"] for row in variant_rows
            ],
            "translation_delta_from_identity_degrees": [
                row["translation_delta_from_identity_degrees"] for row in variant_rows
            ],
            "focal_relative_error_delta_from_identity": [
                row["focal_relative_error_delta_from_identity"] for row in variant_rows
            ],
            "principal_point_normalized_error_delta_from_identity": [
                row["principal_point_normalized_error_delta_from_identity"]
                for row in variant_rows
            ],
        }
        if depth_evaluated:
            metric_values["depth_mean_abs_rel"] = [
                row["depth_mean_abs_rel"] for row in variant_rows
            ]
            metric_values["depth_abs_rel_delta_from_identity"] = [
                row["depth_abs_rel_delta_from_identity"] for row in variant_rows
            ]
        if point_cloud_evaluated:
            metric_values["point_accuracy_mean_meters"] = [
                row["point_accuracy_mean_meters"] for row in variant_rows
            ]
            metric_values["point_completeness_mean_meters"] = [
                row["point_completeness_mean_meters"] for row in variant_rows
            ]
            metric_values["point_accuracy_delta_from_identity_meters"] = [
                row["point_accuracy_delta_from_identity_meters"] for row in variant_rows
            ]
            metric_values["point_completeness_delta_from_identity_meters"] = [
                row["point_completeness_delta_from_identity_meters"]
                for row in variant_rows
            ]
        metric_arrays = {
            label: np.asarray(values, dtype=np.float64)
            for label, values in metric_values.items()
        }
        complete_metrics = {
            label: values
            for label, values in metric_arrays.items()
            if np.isfinite(values).all()
        }
        by_variant[variant] = {
            "scene_count": len(scenes),
            "scenes": scenes,
            "metric_availability": {
                label: {
                    "valid_scene_count": int(np.count_nonzero(np.isfinite(values))),
                    "undefined_scene_count": int(
                        np.count_nonzero(~np.isfinite(values))
                    ),
                    "included_in_scene_bootstrap": label in complete_metrics,
                }
                for label, values in metric_arrays.items()
            },
            "scene_bootstrap": scene_bootstrap_summary(
                complete_metrics,
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
        "point_cloud_evaluated": point_cloud_evaluated,
        "identity": next(iter(identities.values())) if len(identities) == 1 else None,
        "identities": identities,
        "evaluations": rows,
        "by_variant": by_variant,
    }


def summarize_dtu_evaluations(
    paths: list[Path],
    *,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 17,
) -> dict[str, object]:
    """Aggregate paired DTU pose/intrinsics and optional point-map metrics."""

    if not paths:
        raise ValueError("at least one DTU evaluation record is required")
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        point_cloud = record.get("point_cloud")
        intrinsics = record["intrinsics"]
        rows.append(
            {
                "scene": str(record.get("scene", path.parent.name)),
                "variant": str(record.get("variant", Path(record["prediction"]).stem)),
                "rotation_median_degrees": record["relative_rotation_degrees"][
                    "median"
                ],
                "translation_median_degrees": record["translation_direction_degrees"][
                    "median"
                ],
                "focal_relative_error_median": intrinsics["focal_relative_error"][
                    "median"
                ],
                "principal_point_normalized_error_median": intrinsics[
                    "principal_point_normalized_error"
                ]["median"],
                "point_accuracy_mean_millimeters": (
                    point_cloud["accuracy_millimeters"]["mean"]
                    if point_cloud is not None
                    else None
                ),
                "point_completeness_mean_millimeters": (
                    point_cloud["completeness_millimeters"]["mean"]
                    if point_cloud is not None
                    else None
                ),
                "point_cloud_evaluated": point_cloud is not None,
                "source": str(path),
            }
        )

    by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    expected_variants: set[str] | None = None
    for row in rows:
        pair = (str(row["scene"]), str(row["variant"]))
        if pair in seen_pairs:
            raise ValueError(
                f"duplicate DTU scene/variant evaluation: "
                f"scene={pair[0]!r}, variant={pair[1]!r}"
            )
        seen_pairs.add(pair)
        by_scene[pair[0]].append(row)
    for scene, scene_rows in sorted(by_scene.items()):
        variants = {str(row["variant"]) for row in scene_rows}
        if expected_variants is None:
            expected_variants = variants
        elif variants != expected_variants:
            raise ValueError(
                f"incomplete paired DTU design for {scene}: "
                f"missing={sorted(expected_variants - variants)}, "
                f"extra={sorted(variants - expected_variants)}"
            )
        identities = [row for row in scene_rows if row["variant"] == "identity"]
        if len(identities) != 1:
            raise ValueError(f"DTU scene {scene} requires exactly one identity")
        identity = identities[0]
        for row in scene_rows:
            for metric in (
                "rotation_median_degrees",
                "translation_median_degrees",
                "focal_relative_error_median",
                "principal_point_normalized_error_median",
                "point_accuracy_mean_millimeters",
                "point_completeness_mean_millimeters",
            ):
                row[f"{metric}_delta_from_identity"] = _optional_difference(
                    row[metric], identity[metric]
                )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    metric_names = (
        "rotation_median_degrees",
        "translation_median_degrees",
        "focal_relative_error_median",
        "principal_point_normalized_error_median",
        "point_accuracy_mean_millimeters",
        "point_completeness_mean_millimeters",
    )
    by_variant: dict[str, dict[str, object]] = {}
    point_metric_variants: list[str] = []
    for variant, variant_rows in sorted(grouped.items()):
        variant_rows = sorted(variant_rows, key=lambda row: str(row["scene"]))
        scenes = [str(row["scene"]) for row in variant_rows]
        point_modes = {bool(row["point_cloud_evaluated"]) for row in variant_rows}
        if len(point_modes) != 1:
            raise ValueError(
                f"inconsistent DTU point-metric availability for variant {variant}"
            )
        if next(iter(point_modes)):
            point_metric_variants.append(variant)
        values: dict[str, np.ndarray] = {}
        for metric in metric_names:
            values[metric] = np.asarray(
                [
                    float(row[metric]) if row[metric] is not None else np.nan
                    for row in variant_rows
                ]
            )
            values[f"{metric}_delta_from_identity"] = np.asarray(
                [
                    (
                        float(row[f"{metric}_delta_from_identity"])
                        if row[f"{metric}_delta_from_identity"] is not None
                        else np.nan
                    )
                    for row in variant_rows
                ]
            )
        complete = {
            metric: metric_values
            for metric, metric_values in values.items()
            if np.isfinite(metric_values).all()
        }
        by_variant[variant] = {
            "scene_count": len(scenes),
            "scenes": scenes,
            "metric_availability": {
                metric: {
                    "valid_scene_count": int(
                        np.count_nonzero(np.isfinite(metric_values))
                    ),
                    "undefined_scene_count": int(
                        np.count_nonzero(~np.isfinite(metric_values))
                    ),
                    "included_in_scene_bootstrap": metric in complete,
                }
                for metric, metric_values in values.items()
            },
            "scene_bootstrap": scene_bootstrap_summary(
                complete,
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
        "point_metric_protocol": "official_mask_plane_threshold_deterministic_cap",
        "point_metric_variants": point_metric_variants,
        "evaluations": rows,
        "by_variant": by_variant,
    }


def _optional_difference(value: object, reference: object) -> float | None:
    if value is None or reference is None:
        return None
    difference = float(value) - float(reference)
    return difference if np.isfinite(difference) else None
