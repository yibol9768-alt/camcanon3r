"""Training-free selection among canonical-fill repair predictions."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import pairwise_relative_pose_errors
from .repair_evaluation import summarize_repair_evaluations


def _rotation_error(record: Mapping[str, Any]) -> float:
    value = record.get("relative_rotation_degrees")
    if not isinstance(value, Mapping) or value.get("median") is None:
        raise ValueError("repair candidate has no ground-truth rotation median")
    result = float(value["median"])
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("ground-truth rotation median must be finite and non-negative")
    return result


def _prediction_fields(path: Path) -> tuple[np.ndarray, str, float]:
    with np.load(path) as prediction:
        extrinsic = np.asarray(prediction["extrinsic"], dtype=np.float64)
        for field in ("world_points_conf", "depth_conf"):
            if field not in prediction:
                continue
            values = np.asarray(prediction[field], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"native confidence has no finite values: {path}")
            return extrinsic, field, float(np.median(finite))
    raise KeyError(f"prediction has no supported native confidence field: {path}")


def select_repair_candidates(
    candidates: Mapping[str, tuple[Path, Mapping[str, Any]]],
    *,
    candidate_order: Sequence[str],
) -> dict[str, object]:
    """Select consensus/native/oracle candidates with deterministic ties."""

    labels = [str(label) for label in candidate_order]
    if len(labels) < 3 or len(set(labels)) != len(labels):
        raise ValueError("repair consensus requires at least three unique candidates")
    if set(candidates) != set(labels):
        raise ValueError(
            "repair candidate design mismatch: "
            f"expected={labels}, actual={sorted(candidates)}"
        )
    extrinsics: dict[str, np.ndarray] = {}
    confidence_fields: dict[str, str] = {}
    confidences: dict[str, float] = {}
    for label in labels:
        extrinsic, field, confidence = _prediction_fields(candidates[label][0])
        extrinsics[label] = extrinsic
        confidence_fields[label] = field
        confidences[label] = confidence

    pairwise: list[dict[str, object]] = []
    disagreements: dict[str, list[float]] = {label: [] for label in labels}
    for first_index, first in enumerate(labels[:-1]):
        for second in labels[first_index + 1 :]:
            errors = pairwise_relative_pose_errors(
                extrinsics[first], extrinsics[second]
            )
            rotation = float(np.median(errors["rotation_degrees"]))
            if not np.isfinite(rotation):
                raise ValueError("repair pairwise rotation disagreement is undefined")
            disagreements[first].append(rotation)
            disagreements[second].append(rotation)
            pairwise.append(
                {
                    "first": first,
                    "second": second,
                    "rotation_median_degrees": rotation,
                }
            )
    consensus_scores = {
        label: float(np.median(disagreements[label])) for label in labels
    }
    order_index = {label: index for index, label in enumerate(labels)}
    consensus = min(
        labels, key=lambda label: (consensus_scores[label], order_index[label])
    )
    native = min(labels, key=lambda label: (-confidences[label], order_index[label]))
    oracle_errors = {label: _rotation_error(candidates[label][1]) for label in labels}
    oracle = min(labels, key=lambda label: (oracle_errors[label], order_index[label]))
    return {
        "candidate_order": labels,
        "pairwise": pairwise,
        "consensus_scores_rotation_degrees": consensus_scores,
        "native_confidence_fields": confidence_fields,
        "native_confidence_medians": confidences,
        "oracle_rotation_errors_degrees": oracle_errors,
        "selected": {
            "analytic_single_pass": labels[0],
            "consensus": consensus,
            "native_confidence": native,
            "oracle": oracle,
        },
        "ground_truth_used": {
            "analytic_single_pass": False,
            "consensus": False,
            "native_confidence": False,
            "oracle": True,
        },
    }


def summarize_consensus_repair(
    scenes: Mapping[
        str,
        dict[str, object],
    ],
    *,
    candidate_order: Sequence[str],
    minimum_gap: float,
    bootstrap_replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
    recovery_threshold: float,
    clean_relative_threshold: float,
) -> dict[str, object]:
    if not scenes:
        raise ValueError("at least one consensus-repair scene is required")
    selections: dict[str, dict[str, object]] = {}
    method_records: dict[str, dict[str, tuple[Mapping[str, Any], ...]]] = {
        method: {}
        for method in (
            "analytic_single_pass",
            "consensus",
            "native_confidence",
            "oracle",
        )
    }
    for scene in sorted(scenes):
        scene_record = scenes[scene]
        candidates = scene_record["candidates"]
        if not isinstance(candidates, Mapping):
            raise TypeError(f"repair candidates must be a mapping: {scene}")
        selection = select_repair_candidates(
            candidates, candidate_order=candidate_order
        )
        selections[scene] = selection
        selected = selection["selected"]
        assert isinstance(selected, Mapping)
        for method, records in method_records.items():
            label = str(selected[method])
            candidate = candidates[label]
            records[scene] = (
                scene_record["identity"],
                scene_record["corrupt"],
                candidate[1],
                scene_record["clean_control"],
            )

    method_summaries = {
        method: summarize_repair_evaluations(
            records,
            minimum_gap=minimum_gap,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
            recovery_threshold=recovery_threshold,
            clean_relative_threshold=clean_relative_threshold,
        )
        for method, records in method_records.items()
    }
    selection_frequencies = {
        method: dict(
            sorted(
                Counter(
                    str(selections[scene]["selected"][method]) for scene in selections
                ).items()
            )
        )
        for method in method_records
    }
    primary = "relative_rotation_median_degrees"
    consensus_metric = method_summaries["consensus"]["by_metric"][primary]
    analytic_metric = method_summaries["analytic_single_pass"]["by_metric"][primary]
    consensus_repaired = consensus_metric["scene_bootstrap"]["metrics"][
        "repaired_error"
    ]["point_estimate"]
    analytic_repaired = analytic_metric["scene_bootstrap"]["metrics"]["repaired_error"][
        "point_estimate"
    ]
    consensus_gate = consensus_metric["promotion_gate"]
    promotion = {
        "primary_metric": primary,
        "consensus_repaired_error": consensus_repaired,
        "analytic_single_pass_repaired_error": analytic_repaired,
        "beats_single_pass_analytic_repair": consensus_repaired < analytic_repaired,
        "recovery_pass": consensus_gate["point_recovery_pass"],
        "clean_cost_pass": consensus_gate["point_clean_cost_pass"],
    }
    promotion["all_point_estimate_gates_pass"] = bool(
        promotion["beats_single_pass_analytic_repair"]
        and promotion["recovery_pass"]
        and promotion["clean_cost_pass"]
    )
    return {
        "schema_version": "1.0",
        "scene_count": len(scenes),
        "scenes": sorted(scenes),
        "candidate_order": list(candidate_order),
        "selections": selections,
        "selection_frequencies": selection_frequencies,
        "method_summaries": method_summaries,
        "promotion": promotion,
    }


def read_evaluation(path: Path, *, scene: str, variant: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"repair evaluation is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("scene") != scene or record.get("variant") != variant:
        raise ValueError(f"repair evaluation identity mismatch: {path}")
    return record
