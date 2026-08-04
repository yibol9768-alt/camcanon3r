"""Build auditable cross-transform reliability cases without ground-truth scores."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .metrics import aligned_depth_consistency, pairwise_relative_pose_errors

GT_FIELDS = (
    "rotation_median_degrees",
    "translation_median_degrees",
    "focal_relative_error_median",
    "principal_point_normalized_error_median",
    "depth_mean_abs_rel",
    "point_accuracy_mean_meters",
    "point_completeness_mean_meters",
)


def _median_or_none(values: list[float | None]) -> float | None:
    finite = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    return float(np.median(finite)) if len(finite) else None


def _native_confidence(path: Path) -> tuple[str, float]:
    with np.load(path) as prediction:
        for field in ("world_points_conf", "depth_conf"):
            if field not in prediction:
                continue
            values = np.asarray(prediction[field], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"native confidence has no finite values: {path}")
            return field, float(np.median(finite))
    raise KeyError(f"prediction has no supported native confidence field: {path}")


def _scene_predictions(
    prediction_root: Path, variants: tuple[str, ...]
) -> dict[str, dict[str, Path]]:
    expected = set(variants)
    scenes: dict[str, dict[str, Path]] = {}
    for scene_dir in sorted(path for path in prediction_root.iterdir() if path.is_dir()):
        records = {path.stem: path for path in scene_dir.glob("*.npz")}
        if set(records) != expected:
            raise ValueError(
                f"prediction design mismatch for {scene_dir.name}: "
                f"missing={sorted(expected - set(records))}, "
                f"extra={sorted(set(records) - expected)}"
            )
        scenes[scene_dir.name] = records
    if not scenes:
        raise ValueError("prediction root contains no scene directories")
    return scenes


def _validate_evaluation_design(
    result_root: Path,
    scenes: set[str],
    variants: tuple[str, ...],
) -> None:
    result_scenes = {
        path.name for path in result_root.iterdir() if path.is_dir()
    }
    if result_scenes != scenes:
        raise ValueError(
            "evaluation scene design mismatch: "
            f"missing={sorted(scenes - result_scenes)}, "
            f"extra={sorted(result_scenes - scenes)}"
        )
    expected = {f"{variant}_vs_gt" for variant in variants}
    for scene in sorted(scenes):
        records = {
            path.stem for path in (result_root / scene).glob("*_vs_gt.json")
        }
        if records != expected:
            raise ValueError(
                f"evaluation variant design mismatch for {scene}: "
                f"missing={sorted(expected - records)}, "
                f"extra={sorted(records - expected)}"
            )


def _pairwise_disagreement(first: Path, second: Path) -> dict[str, float]:
    with np.load(first) as first_data:
        first_extrinsic = first_data["extrinsic"]
        first_depth = first_data["depth"]
        first_affine = first_data["source_to_model_affine"]
    with np.load(second) as second_data:
        second_extrinsic = second_data["extrinsic"]
        second_depth = second_data["depth"]
        second_affine = second_data["source_to_model_affine"]

    pose = pairwise_relative_pose_errors(first_extrinsic, second_extrinsic)
    first_to_second = aligned_depth_consistency(
        first_depth,
        second_depth,
        first_affine,
        second_affine,
    )
    second_to_first = aligned_depth_consistency(
        second_depth,
        first_depth,
        second_affine,
        first_affine,
    )
    return {
        "rotation": float(np.median(pose["rotation_degrees"])),
        "translation": float(
            np.median(pose["translation_direction_degrees"])
        ),
        "depth": float(
            np.mean(
                [
                    first_to_second["mean_abs_rel"],
                    second_to_first["mean_abs_rel"],
                ]
            )
        ),
    }


def _evaluation_record(
    result_root: Path, scene: str, variant: str
) -> tuple[Path, dict[str, object]]:
    path = result_root / scene / f"{variant}_vs_gt.json"
    if not path.is_file():
        raise FileNotFoundError(f"ground-truth evaluation is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("scene") != scene or record.get("variant") != variant:
        raise ValueError(f"evaluation identity mismatch: {path}")
    return path, record


def build_reliability_cases(
    prediction_root: Path,
    result_root: Path,
    *,
    variants: tuple[str, ...],
    model: str,
    dataset: str,
) -> dict[str, object]:
    """Build one case per scene/variant using leave-one-transform-out scores.

    For every candidate transform, disagreement is the median of its pairwise
    disagreement with every other registered transform.  Identity is therefore
    treated as one candidate rather than a privileged zero-score reference.
    Ground truth is copied only after the score has been computed and never
    enters score construction.
    """

    if len(variants) < 3 or len(set(variants)) != len(variants):
        raise ValueError("reliability design requires at least three unique variants")
    scenes = _scene_predictions(prediction_root, variants)
    _validate_evaluation_design(result_root, set(scenes), variants)
    cases: list[dict[str, object]] = []
    for scene, predictions in scenes.items():
        disagreements: dict[str, dict[str, list[float | None]]] = {
            variant: {"rotation": [], "translation": [], "depth": []}
            for variant in variants
        }
        for first, second in combinations(variants, 2):
            values = _pairwise_disagreement(
                predictions[first], predictions[second]
            )
            for variant in (first, second):
                for metric, value in values.items():
                    disagreements[variant][metric].append(value)

        for variant in variants:
            evaluation_path, evaluation = _evaluation_record(
                result_root, scene, variant
            )
            confidence_field, confidence = _native_confidence(predictions[variant])
            gt = {field: evaluation.get(field) for field in GT_FIELDS}
            if gt["rotation_median_degrees"] is None:
                raise ValueError(f"rotation GT is undefined: {evaluation_path}")
            cases.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "scene": scene,
                    "variant": variant,
                    "anchor_count": len(variants) - 1,
                    "scores": {
                        "rotation_disagreement_degrees": _median_or_none(
                            disagreements[variant]["rotation"]
                        ),
                        "translation_disagreement_degrees": _median_or_none(
                            disagreements[variant]["translation"]
                        ),
                        "depth_disagreement_abs_rel": _median_or_none(
                            disagreements[variant]["depth"]
                        ),
                        "native_confidence_field": confidence_field,
                        "native_confidence_median": confidence,
                        "native_uncertainty": -confidence,
                    },
                    "ground_truth": gt,
                    "prediction": str(predictions[variant].resolve()),
                    "evaluation": str(evaluation_path.resolve()),
                }
            )
    return {
        "schema_version": "1.0",
        "dataset": dataset,
        "model": model,
        "scene_count": len(scenes),
        "variant_count": len(variants),
        "case_count": len(cases),
        "variants": list(variants),
        "score_protocol": {
            "cross_transform": "candidate median across all other transforms",
            "pairwise_depth": (
                "mean of both directed, scale-aligned depth AbsRel values"
            ),
            "native_uncertainty": "negative median native confidence",
            "ground_truth_used_in_score": False,
        },
        "cases": cases,
    }
