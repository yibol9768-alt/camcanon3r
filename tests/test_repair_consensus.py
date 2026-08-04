from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from camcanon3r.repair_consensus import (
    _prediction_compute,
    select_repair_candidates,
    summarize_consensus_repair,
)

ORDER = ("neutral_gray", "black", "image_mean")


def _prediction(path: Path, rotation_degrees: float, confidence: float) -> None:
    extrinsic = np.repeat(np.eye(4)[None], 2, axis=0)
    extrinsic[1, :3, :3] = Rotation.from_euler(
        "z", rotation_degrees, degrees=True
    ).as_matrix()
    extrinsic[1, 0, 3] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        extrinsic=extrinsic,
        world_points_conf=np.full((2, 2, 2), confidence),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "inference_seconds": 0.5 + rotation_degrees / 100.0,
                "load_seconds": 2.0,
                "model_reused_across_variants": True,
                "peak_vram_bytes": 1024,
            }
        ),
        encoding="utf-8",
    )


def _evaluation(rotation: float) -> dict[str, object]:
    return {
        "prediction": f"prediction-{rotation}",
        "relative_rotation_degrees": {"median": rotation},
        "translation_direction_degrees": {"median": rotation * 2.0},
        "depth": {"mean_abs_rel": rotation / 100.0},
    }


def test_prediction_compute_supports_dust3r_pairwise_and_alignment(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "dust3r.npz"
    prediction.touch()
    prediction.with_suffix(".json").write_text(
        json.dumps(
            {
                "pairwise_inference_seconds": 1.25,
                "alignment_seconds": 3.75,
                "load_seconds": 4.0,
                "model_reused_across_variants": True,
                "peak_vram_bytes": 2048,
            }
        ),
        encoding="utf-8",
    )
    compute = _prediction_compute(prediction)
    assert compute["model_compute_seconds"] == pytest.approx(5.0)
    assert compute["components"] == {
        "pairwise_inference_seconds": 1.25,
        "alignment_seconds": 3.75,
    }
    assert compute["peak_vram_bytes"] == 2048


def test_consensus_native_and_oracle_use_separate_frozen_signals(
    tmp_path: Path,
) -> None:
    candidates = {}
    for label, rotation, confidence, gt_error in (
        ("neutral_gray", 0.0, 0.7, 1.0),
        ("black", 5.0, 0.8, 3.0),
        ("image_mean", 6.0, 0.9, 2.0),
    ):
        path = tmp_path / f"{label}.npz"
        _prediction(path, rotation, confidence)
        candidates[label] = (path, _evaluation(gt_error))

    result = select_repair_candidates(candidates, candidate_order=ORDER)
    assert result["selected"] == {
        "analytic_single_pass": "neutral_gray",
        "consensus": "black",
        "native_confidence": "image_mean",
        "oracle": "neutral_gray",
    }
    assert result["ground_truth_used"]["consensus"] is False
    assert result["ground_truth_used"]["oracle"] is True
    assert result["consensus_scores_rotation_degrees"]["black"] == pytest.approx(3.0)

    candidates["neutral_gray"] = (
        candidates["neutral_gray"][0],
        _evaluation(100.0),
    )
    changed = select_repair_candidates(candidates, candidate_order=ORDER)
    assert changed["selected"]["consensus"] == "black"
    assert changed["selected"]["oracle"] == "image_mean"


def test_consensus_summary_keeps_methods_and_models_unpooled(tmp_path: Path) -> None:
    scenes = {}
    for scene_index, scene in enumerate(("first", "second", "third")):
        candidates = {}
        for label, rotation, confidence, gt_offset in (
            ("neutral_gray", 0.0, 0.7, 2.0),
            ("black", 5.0, 0.8, 1.0),
            ("image_mean", 6.0, 0.9, 3.0),
        ):
            path = tmp_path / scene / f"{label}.npz"
            _prediction(path, rotation, confidence)
            candidates[label] = (
                path,
                _evaluation(gt_offset + 0.1 * scene_index),
            )
        scenes[scene] = {
            "identity": _evaluation(1.0 + 0.1 * scene_index),
            "corrupt": _evaluation(5.0 + 0.1 * scene_index),
            "clean_control": _evaluation(1.01 + 0.101 * scene_index),
            "candidates": candidates,
        }
    summary = summarize_consensus_repair(
        scenes,
        candidate_order=ORDER,
        minimum_gap=1e-12,
        bootstrap_replicates=200,
        confidence_level=0.95,
        bootstrap_seed=17,
        recovery_threshold=0.30,
        clean_relative_threshold=0.02,
    )
    assert summary["selection_frequencies"]["consensus"] == {"black": 3}
    assert summary["selection_frequencies"]["native_confidence"] == {"image_mean": 3}
    assert summary["promotion"]["beats_single_pass_analytic_repair"] is True
    assert summary["promotion"]["all_point_estimate_gates_pass"] is True
    assert (
        summary["method_compute"]["analytic_single_pass"]["model_runs_per_scene"] == 1
    )
    assert summary["method_compute"]["consensus"]["model_runs_per_scene"] == 3
    assert (
        summary["candidate_compute"]["neutral_gray"]["maximum_peak_vram_bytes"] == 1024
    )
