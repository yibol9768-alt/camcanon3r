"""Cross-dataset analysis for the support-preserving letterbox control."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .summary import scene_bootstrap_summary
from .support_control import SUPPORT_VARIANTS

METRICS = {
    "rotation_degrees": {
        "eth3d": "rotation_median_degrees",
        "dtu": "rotation_median_degrees",
        "eth3d_scale": 1.0,
        "dtu_scale": 1.0,
        "unit": "degrees",
    },
    "translation_degrees": {
        "eth3d": "translation_median_degrees",
        "dtu": "translation_median_degrees",
        "eth3d_scale": 1.0,
        "dtu_scale": 1.0,
        "unit": "degrees",
    },
    "focal_percentage_points": {
        "eth3d": "focal_relative_error_median",
        "dtu": "focal_relative_error_median",
        "eth3d_scale": 100.0,
        "dtu_scale": 100.0,
        "unit": "percentage_points",
    },
    "principal_point_percent_diagonal": {
        "eth3d": "principal_point_normalized_error_median",
        "dtu": "principal_point_normalized_error_median",
        "eth3d_scale": 100.0,
        "dtu_scale": 100.0,
        "unit": "percent_image_diagonal",
    },
    "depth_abs_rel_percentage_points": {
        "eth3d": "depth_mean_abs_rel",
        "dtu": None,
        "eth3d_scale": 100.0,
        "dtu_scale": 1.0,
        "unit": "percentage_points",
    },
    "point_accuracy_millimeters": {
        "eth3d": "point_accuracy_mean_meters",
        "dtu": "point_accuracy_mean_millimeters",
        "eth3d_scale": 1000.0,
        "dtu_scale": 1.0,
        "unit": "millimeters",
    },
    "point_completeness_millimeters": {
        "eth3d": "point_completeness_mean_meters",
        "dtu": "point_completeness_mean_millimeters",
        "eth3d_scale": 1000.0,
        "dtu_scale": 1.0,
        "unit": "millimeters",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema(summary: dict[str, Any]) -> str:
    if "depth_evaluated" in summary:
        return "eth3d"
    if "point_metric_protocol" in summary:
        return "dtu"
    raise ValueError("unsupported support-control summary schema")


def _finite_or_nan(value: object, scale: float) -> float:
    if value is None:
        return float("nan")
    result = float(value) * scale
    return result if math.isfinite(result) else float("nan")


def analyze_one(
    model: str,
    dataset: str,
    path: Path,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    schema = _schema(summary)
    expected_scenes = 13 if dataset == "eth3d" else 22
    if schema != dataset or int(summary.get("scene_count", -1)) != expected_scenes:
        raise ValueError(f"support-control summary dataset mismatch: {path}")
    rows = summary.get("evaluations")
    if not isinstance(rows, list) or len(rows) != expected_scenes * len(
        SUPPORT_VARIANTS
    ):
        raise ValueError(f"support-control evaluation count mismatch: {path}")
    by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        scene = str(row["scene"])
        variant = str(row["variant"])
        if variant not in SUPPORT_VARIANTS or variant in by_scene.setdefault(scene, {}):
            raise ValueError(f"support-control scene/variant design drift: {path}")
        by_scene[scene][variant] = row
    if len(by_scene) != expected_scenes or any(
        set(records) != set(SUPPORT_VARIANTS) for records in by_scene.values()
    ):
        raise ValueError(f"support-control paired design is incomplete: {path}")
    scenes = sorted(by_scene)
    anchor = SUPPORT_VARIANTS[0]
    by_variant: dict[str, object] = {}
    for variant in SUPPORT_VARIANTS:
        values: dict[str, np.ndarray] = {}
        availability: dict[str, object] = {}
        for label, spec in METRICS.items():
            field = spec[schema]
            if field is None:
                continue
            scale = float(spec[f"{schema}_scale"])
            absolute = np.asarray(
                [
                    _finite_or_nan(by_scene[scene][variant].get(field), scale)
                    for scene in scenes
                ],
                dtype=np.float64,
            )
            anchor_values = np.asarray(
                [
                    _finite_or_nan(by_scene[scene][anchor].get(field), scale)
                    for scene in scenes
                ],
                dtype=np.float64,
            )
            delta = absolute - anchor_values
            availability[label] = {
                "valid_scene_count": int(np.count_nonzero(np.isfinite(absolute))),
                "paired_valid_scene_count": int(np.count_nonzero(np.isfinite(delta))),
                "included_in_scene_bootstrap": bool(np.isfinite(absolute).all()),
            }
            if np.isfinite(absolute).all():
                values[f"{label}_absolute"] = absolute
            if np.isfinite(delta).all():
                values[f"{label}_delta_from_symmetric_letterbox"] = delta
        by_variant[variant] = {
            "scene_count": len(scenes),
            "metric_availability": availability,
            "scene_bootstrap": scene_bootstrap_summary(
                values,
                scenes=scenes,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            ),
        }
    return {
        "model": model,
        "dataset": dataset,
        "scene_count": len(scenes),
        "variants": list(SUPPORT_VARIANTS),
        "source": str(path),
        "source_sha256": _sha256(path),
        "by_variant": by_variant,
    }


def analyze_support_control(
    summaries: list[tuple[str, str, Path]],
    protocol_path: Path,
    *,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 17,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    gate = protocol["promotion_gate"]
    expected_models = [str(value) for value in gate["required_models"]]
    expected_datasets = [str(value) for value in gate["required_datasets"]]
    expected = {
        (model, dataset) for model in expected_models for dataset in expected_datasets
    }
    supplied = {(model, dataset) for model, dataset, _ in summaries}
    if len(supplied) != len(summaries) or supplied != expected:
        raise ValueError(
            "support-control summaries do not match the frozen model/dataset design"
        )
    analyses = [
        analyze_one(
            model,
            dataset,
            path,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
        for model, dataset, path in summaries
    ]
    primary = str(protocol["primary_variant"])
    threshold = float(gate["minimum_rotation_delta_degrees"])
    support: list[dict[str, object]] = []
    for record in analyses:
        metric = record["by_variant"][primary]["scene_bootstrap"]["metrics"][
            "rotation_degrees_delta_from_symmetric_letterbox"
        ]
        support.append(
            {
                "model": record["model"],
                "dataset": record["dataset"],
                "rotation_delta": metric,
                "crosses_point_estimate_threshold": (
                    float(metric["point_estimate"]) > threshold
                ),
            }
        )
    return {
        "schema_version": "support-control-analysis-1.0",
        "status": "complete",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "models_are_separate": True,
        "datasets_are_separate": True,
        "metric_units": {label: spec["unit"] for label, spec in METRICS.items()},
        "analyses": analyses,
        "promotion_gate": {
            "variant": primary,
            "rotation_delta_threshold_degrees": threshold,
            "support": support,
            "passes_all_models_and_datasets": all(
                bool(record["crosses_point_estimate_threshold"]) for record in support
            ),
        },
        "interpretation": {
            "all_source_rgb_preserved": True,
            "same_scale_canvas_and_padding_count": True,
            "failed_gate_remains_negative_result": True,
        },
    }
