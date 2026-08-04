"""Paired severity and crop-scope analysis for frozen mechanism sweeps."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

from .prediction import write_json_atomic
from .statistics import scene_bootstrap_summary

_RAW_METRICS = (
    "rotation_median_degrees",
    "translation_median_degrees",
    "focal_relative_error_median",
    "principal_point_normalized_error_median",
    "depth_mean_abs_rel",
)
_FAMILIES = {
    "independent_asymmetric_crop": (
        "asymmetric_crop_090",
        "asymmetric_crop_075",
        "asymmetric_crop_060",
    ),
    "shared_asymmetric_crop": (
        "shared_asymmetric_crop_090",
        "shared_asymmetric_crop_075",
        "shared_asymmetric_crop_060",
    ),
    "center_crop": (
        "center_crop_090",
        "center_crop_075",
        "center_crop_060",
    ),
    "letterbox": ("letterbox_square",),
}
_EXPECTED_VARIANTS = {"identity"}.union(
    *(set(family_variants) for family_variants in _FAMILIES.values())
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_array(values: Sequence[object], label: str) -> np.ndarray:
    array = np.asarray(
        [float(value) if value is not None else np.nan for value in values],
        dtype=np.float64,
    )
    if np.isinf(array).any():
        raise ValueError(f"mechanism metric contains infinity: {label}")
    return array


def _analyze_one(
    summary: Mapping[str, object],
    *,
    variants: Sequence[str],
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
    rotation_threshold: float,
    depth_threshold: float,
) -> dict[str, object]:
    rows = summary.get("evaluations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("mechanism summary has no scene evaluations")
    grouped: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("mechanism evaluation row must be a mapping")
        scene = str(row["scene"])
        variant = str(row["variant"])
        if variant in grouped[scene]:
            raise ValueError(f"duplicate mechanism scene/variant: {scene}/{variant}")
        grouped[scene][variant] = row
    expected = set(variants)
    for scene, scene_rows in grouped.items():
        if set(scene_rows) != expected:
            raise ValueError(
                f"mechanism variant design mismatch for {scene}: "
                f"missing={sorted(expected - set(scene_rows))}, "
                f"extra={sorted(set(scene_rows) - expected)}"
            )
        if "identity" not in scene_rows:
            raise ValueError(f"mechanism scene has no identity: {scene}")
    scenes = sorted(grouped)

    by_variant: dict[str, dict[str, object]] = {}
    delta_arrays: dict[str, dict[str, np.ndarray]] = {}
    for variant in variants:
        raw_arrays: dict[str, np.ndarray] = {}
        candidate_deltas: dict[str, np.ndarray] = {}
        metric_availability: dict[str, dict[str, object]] = {}
        for metric in _RAW_METRICS:
            raw = _metric_array(
                [grouped[scene][variant].get(metric) for scene in scenes],
                f"{variant}/{metric}",
            )
            identity = _metric_array(
                [grouped[scene]["identity"].get(metric) for scene in scenes],
                f"identity/{metric}",
            )
            paired_finite = np.isfinite(raw) & np.isfinite(identity)
            included = bool(paired_finite.all())
            metric_availability[metric] = {
                "candidate_valid_scene_count": int(
                    np.count_nonzero(np.isfinite(raw))
                ),
                "identity_valid_scene_count": int(
                    np.count_nonzero(np.isfinite(identity))
                ),
                "paired_valid_scene_count": int(np.count_nonzero(paired_finite)),
                "undefined_scene_count": int(
                    len(scenes) - np.count_nonzero(paired_finite)
                ),
                "included_in_scene_bootstrap": included,
            }
            if not included:
                continue
            raw_arrays[metric] = raw
            candidate_deltas[f"{metric}_delta_from_identity"] = raw - identity
        complete = {**raw_arrays, **candidate_deltas}
        bootstrap = scene_bootstrap_summary(
            complete,
            scenes=scenes,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        bootstrap["metric_availability"] = metric_availability
        by_variant[variant] = bootstrap
        delta_arrays[variant] = candidate_deltas

    contrasts: dict[str, dict[str, object]] = {}
    for fraction in ("090", "075", "060"):
        for comparison, first, second in (
            (
                "independent_minus_shared",
                f"asymmetric_crop_{fraction}",
                f"shared_asymmetric_crop_{fraction}",
            ),
            (
                "independent_minus_center",
                f"asymmetric_crop_{fraction}",
                f"center_crop_{fraction}",
            ),
            (
                "shared_minus_center",
                f"shared_asymmetric_crop_{fraction}",
                f"center_crop_{fraction}",
            ),
        ):
            metrics = {}
            for metric in _RAW_METRICS:
                field = f"{metric}_delta_from_identity"
                if field in delta_arrays[first] and field in delta_arrays[second]:
                    metrics[field] = (
                        delta_arrays[first][field] - delta_arrays[second][field]
                    )
            contrasts[f"{comparison}_{fraction}"] = scene_bootstrap_summary(
                metrics,
                scenes=scenes,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )

    family_gates: dict[str, dict[str, object]] = {}
    for family, family_variants in _FAMILIES.items():
        rotation_crossings = []
        depth_crossings = []
        rotation_values = []
        for variant in family_variants:
            rotation_field = "rotation_median_degrees_delta_from_identity"
            if rotation_field not in delta_arrays[variant]:
                raise ValueError(
                    f"registered rotation metric is incomplete: {variant}"
                )
            rotation = float(
                np.median(delta_arrays[variant][rotation_field])
            )
            rotation_values.append(rotation)
            if rotation > rotation_threshold:
                rotation_crossings.append(variant)
            depth_field = "depth_mean_abs_rel_delta_from_identity"
            if depth_field in delta_arrays[variant]:
                depth = float(np.median(delta_arrays[variant][depth_field]))
                if depth > depth_threshold:
                    depth_crossings.append(variant)
        family_gates[family] = {
            "variants": list(family_variants),
            "rotation_delta_point_estimates": rotation_values,
            "rotation_monotone_as_retention_decreases": (
                all(
                    first <= second
                    for first, second in pairwise(rotation_values)
                )
                if len(rotation_values) > 1
                else None
            ),
            "rotation_crossing_variants": rotation_crossings,
            "depth_crossing_variants": depth_crossings,
            "crosses_registered_threshold": bool(
                rotation_crossings or depth_crossings
            ),
        }
    return {
        "scene_count": len(scenes),
        "scenes": scenes,
        "variant_count": len(variants),
        "variants": list(variants),
        "by_variant": by_variant,
        "paired_contrasts": contrasts,
        "family_gates": family_gates,
    }


def analyze_mechanism_summaries(
    records: Sequence[tuple[str, str, Path]],
    variant_config_path: Path,
    *,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 17,
    rotation_threshold: float = 2.0,
    depth_threshold: float = 0.05,
    output_path: Path | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("at least one mechanism summary is required")
    config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    if not config.get("frozen_before_benchmark_scale_mechanism_results"):
        raise ValueError("mechanism variant config is not frozen")
    variants = [str(value) for value in config["ordered_variants"]]
    if len(variants) != len(_EXPECTED_VARIANTS) or set(variants) != _EXPECTED_VARIANTS:
        raise ValueError("mechanism variant config does not match frozen families")
    analyses: dict[str, dict[str, object]] = {}
    seen: set[tuple[str, str]] = set()
    for model, dataset, path in records:
        identity = (str(model), str(dataset))
        if identity in seen:
            raise ValueError(f"duplicate mechanism model/dataset: {identity}")
        seen.add(identity)
        summary = json.loads(path.read_text(encoding="utf-8"))
        key = f"{model}/{dataset}"
        analyses[key] = {
            "model": str(model),
            "dataset": str(dataset),
            "summary": str(path),
            "summary_sha256": _sha256(path),
            "analysis": _analyze_one(
                summary,
                variants=variants,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed,
                rotation_threshold=rotation_threshold,
                depth_threshold=depth_threshold,
            ),
        }
    datasets = sorted(
        {str(record["dataset"]) for record in analyses.values()}
    )
    family_support = {}
    for family in _FAMILIES:
        crossing = [
            key
            for key, record in analyses.items()
            if record["analysis"]["family_gates"][family][
                "crosses_registered_threshold"
            ]
        ]
        evaluated_models_by_dataset = {
            dataset: sorted(
                str(record["model"])
                for record in analyses.values()
                if record["dataset"] == dataset
            )
            for dataset in datasets
        }
        crossing_models_by_dataset = {
            dataset: sorted(
                str(analyses[key]["model"])
                for key in crossing
                if analyses[key]["dataset"] == dataset
            )
            for dataset in datasets
        }
        datasets_with_all_models_crossing = [
            dataset
            for dataset in datasets
            if crossing_models_by_dataset[dataset]
            == evaluated_models_by_dataset[dataset]
        ]
        family_support[family] = {
            "crossing_model_datasets": crossing,
            "crossing_count": len(crossing),
            "evaluated_count": len(analyses),
            "crosses_in_every_evaluated_model_dataset": len(crossing)
            == len(analyses),
            "evaluated_models_by_dataset": evaluated_models_by_dataset,
            "crossing_models_by_dataset": crossing_models_by_dataset,
            "datasets_with_all_evaluated_models_crossing": (
                datasets_with_all_models_crossing
            ),
            "meets_two_dataset_gate": len(datasets_with_all_models_crossing) >= 2,
        }
    families_meeting_two_dataset_gate = [
        family
        for family, support in family_support.items()
        if support["meets_two_dataset_gate"]
    ]
    report = {
        "schema_version": "1.0",
        "variant_config": str(variant_config_path),
        "variant_config_sha256": _sha256(variant_config_path),
        "rotation_threshold_degrees": rotation_threshold,
        "depth_abs_rel_threshold": depth_threshold,
        "bootstrap_replicates": bootstrap_replicates,
        "confidence_level": confidence_level,
        "bootstrap_seed": bootstrap_seed,
        "analyses": analyses,
        "family_support": family_support,
        "hypothesis_gate": {
            "required_family_count": 2,
            "required_dataset_count_per_family": 2,
            "evaluated_datasets": datasets,
            "families_meeting_two_dataset_gate": (
                families_meeting_two_dataset_gate
            ),
            "meets_two_family_two_dataset_gate": (
                len(families_meeting_two_dataset_gate) >= 2
            ),
            "conservative_rule": (
                "A dataset counts for a family only when every supplied model "
                "on that dataset crosses a registered point-estimate threshold."
            ),
        },
    }
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
