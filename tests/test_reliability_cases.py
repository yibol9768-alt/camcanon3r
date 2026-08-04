from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from camcanon3r.reliability_cases import build_reliability_cases

VARIANTS = ("identity", "center_crop_075", "asymmetric_crop_075")


def _prediction(path: Path, angle: float, confidence: float) -> None:
    extrinsic = np.repeat(np.eye(4)[None], 2, axis=0)
    cosine, sine = np.cos(angle), np.sin(angle)
    extrinsic[1, :3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    extrinsic[1, 0, 3] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        extrinsic=extrinsic,
        depth=np.full((2, 3, 4), 1.0 + angle),
        source_to_model_affine=np.repeat(np.eye(3)[None], 2, axis=0),
        world_points_conf=np.full((2, 3, 4), confidence),
    )


def _evaluation(path: Path, scene: str, variant: str, error: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scene": scene,
                "variant": variant,
                "rotation_median_degrees": error,
                "translation_median_degrees": error,
                "focal_relative_error_median": 0.1,
                "principal_point_normalized_error_median": 0.01,
                "depth_mean_abs_rel": error / 100.0,
                "point_accuracy_mean_meters": 0.2,
                "point_completeness_mean_meters": 0.3,
            }
        ),
        encoding="utf-8",
    )


def _design(tmp_path: Path) -> tuple[Path, Path]:
    predictions = tmp_path / "predictions"
    results = tmp_path / "results"
    for scene in ("first", "second"):
        for index, variant in enumerate(VARIANTS):
            _prediction(
                predictions / scene / f"{variant}.npz",
                angle=0.05 * index,
                confidence=10.0 - index,
            )
            _evaluation(
                results / scene / f"{variant}_vs_gt.json",
                scene,
                variant,
                error=float(index),
            )
    return predictions, results


def test_build_reliability_cases_is_complete_and_unprivileged(
    tmp_path: Path,
) -> None:
    predictions, results = _design(tmp_path)
    output = build_reliability_cases(
        predictions,
        results,
        variants=VARIANTS,
        model="test-model",
        dataset="test-data",
    )

    assert output["scene_count"] == 2
    assert output["case_count"] == 6
    cases = output["cases"]
    assert {case["anchor_count"] for case in cases} == {2}
    identity = next(case for case in cases if case["variant"] == "identity")
    assert identity["scores"]["rotation_disagreement_degrees"] > 0.0
    assert identity["scores"]["native_uncertainty"] == -10.0
    assert output["score_protocol"]["ground_truth_used_in_score"] is False

    reversed_output = build_reliability_cases(
        predictions,
        results,
        variants=tuple(reversed(VARIANTS)),
        model="test-model",
        dataset="test-data",
    )
    first_scores = {
        case["variant"]: case["scores"]["depth_disagreement_abs_rel"]
        for case in cases
    }
    reversed_scores = {
        case["variant"]: case["scores"]["depth_disagreement_abs_rel"]
        for case in reversed_output["cases"]
    }
    assert reversed_scores == pytest.approx(first_scores)


def test_build_reliability_cases_rejects_extra_predictions(tmp_path: Path) -> None:
    predictions, results = _design(tmp_path)
    _prediction(predictions / "first" / "extra.npz", angle=0.0, confidence=1.0)
    with pytest.raises(ValueError, match="prediction design mismatch"):
        build_reliability_cases(
            predictions,
            results,
            variants=VARIANTS,
            model="test-model",
            dataset="test-data",
        )


def test_build_reliability_cases_rejects_extra_evaluation(tmp_path: Path) -> None:
    predictions, results = _design(tmp_path)
    _evaluation(results / "first" / "extra_vs_gt.json", "first", "extra", 1.0)
    with pytest.raises(ValueError, match="evaluation variant design mismatch"):
        build_reliability_cases(
            predictions,
            results,
            variants=VARIANTS,
            model="test-model",
            dataset="test-data",
        )
