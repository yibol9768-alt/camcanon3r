#!/usr/bin/env python3
"""Audit final DTU claim gates from frozen, provenance-bound evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

RELIABILITY_VARIANTS = (
    "identity",
    "center_crop_075",
    "asymmetric_crop_075",
    "letterbox_square",
)
ROTATION_REPAIR_METRIC = "relative_rotation_median_degrees"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument(
        "--reliability",
        action="append",
        nargs=3,
        metavar=("MODEL", "DISAGREEMENT", "NATIVE"),
        required=True,
    )
    parser.add_argument(
        "--repair",
        action="append",
        nargs=2,
        metavar=("MODEL", "REPORT"),
        required=True,
    )
    parser.add_argument("--expected-models", nargs="+", default=["vggt", "dust3r"])
    parser.add_argument("--detector-auroc-threshold", type=float, default=0.75)
    parser.add_argument("--repair-recovery-threshold", type=float, default=0.30)
    parser.add_argument("--clean-relative-threshold", type=float, default=0.02)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"final-claim source is not a JSON object: {path}")
    return payload


def _resolve_source(reference: object, report_path: Path) -> Path:
    path = Path(str(reference))
    candidates = (
        path,
        Path.cwd() / path,
        report_path.resolve().parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"final-claim provenance source is missing: {path}")


def _audit_mechanism(path: Path, *, expected_models: list[str]) -> dict[str, object]:
    report = _read(path)
    analyses = report.get("analyses")
    if not isinstance(analyses, dict):
        raise TypeError("mechanism report contains no analyses")
    expected_keys = {
        f"{model}/{dataset}"
        for model in expected_models
        for dataset in ("eth3d", "dtu")
    }
    if set(analyses) != expected_keys:
        raise ValueError(
            "final mechanism design mismatch: "
            f"expected={sorted(expected_keys)}, actual={sorted(analyses)}"
        )
    for key, record in analyses.items():
        dataset = str(record["dataset"])
        analysis = record["analysis"]
        expected_scenes = 13 if dataset == "eth3d" else 22
        if (
            int(analysis["scene_count"]) != expected_scenes
            or int(analysis["variant_count"]) != 11
        ):
            raise ValueError(f"final mechanism analysis is incomplete: {key}")
    gate = report["hypothesis_gate"]
    if (
        int(gate["required_family_count"]) != 2
        or int(gate["required_dataset_count_per_family"]) != 2
        or set(gate["evaluated_datasets"]) != {"eth3d", "dtu"}
    ):
        raise ValueError("final mechanism hypothesis gate has drifted")
    return {
        "source": str(path),
        "source_sha256": _sha256(path),
        "models": expected_models,
        "datasets": ["eth3d", "dtu"],
        "family_support": report["family_support"],
        "meets_two_family_two_dataset_gate": bool(
            gate["meets_two_family_two_dataset_gate"]
        ),
    }


def _auroc(report: dict[str, Any]) -> dict[str, object]:
    record = report["auroc"]
    status = str(record["status"])
    if status == "defined":
        return {
            "status": status,
            "point_estimate": float(record["point_estimate"]),
            "lower": float(record["lower"]),
            "upper": float(record["upper"]),
            "valid_replicates": int(record["valid_replicates"]),
            "undefined_replicates": int(record["undefined_replicates"]),
        }
    return {
        "status": status,
        "point_estimate": None,
        "lower": None,
        "upper": None,
        "valid_replicates": int(record.get("valid_replicates", 0)),
        "undefined_replicates": int(record.get("undefined_replicates", 0)),
    }


def _audit_reliability_pair(
    model: str,
    disagreement_path: Path,
    native_path: Path,
    *,
    auroc_threshold: float,
) -> dict[str, object]:
    disagreement = _read(disagreement_path)
    native = _read(native_path)
    expected_failure = {"error_operator": ">", "threshold": 2.0}
    for label, report, field in (
        (
            "disagreement",
            disagreement,
            "scores.rotation_disagreement_degrees",
        ),
        ("native", native, "scores.native_uncertainty"),
    ):
        if (
            report.get("failure_definition") != expected_failure
            or int(report.get("case_count", -1)) != 88
            or int(report.get("scene_count", -1)) != 22
            or report.get("input", {}).get("error_field")
            != "ground_truth.rotation_median_degrees"
            or report.get("input", {}).get("uncertainty_field") != field
        ):
            raise ValueError(f"held-out {label} reliability design mismatch: {model}")
        bootstrap = report.get("bootstrap", {})
        if (
            bootstrap.get("resampling_unit") != "scene_cluster"
            or int(bootstrap.get("replicates", -1)) != 10_000
            or int(bootstrap.get("seed", -1)) != 17
            or float(bootstrap.get("confidence_level", -1.0)) != 0.95
        ):
            raise ValueError(f"held-out {label} bootstrap design mismatch: {model}")

    disagreement_input = disagreement["input"]
    native_input = native["input"]
    if disagreement_input.get("cases") != native_input.get(
        "cases"
    ) or disagreement_input.get("cases_sha256") != native_input.get("cases_sha256"):
        raise ValueError(f"held-out reliability methods use different cases: {model}")
    cases_path = _resolve_source(disagreement_input["cases"], disagreement_path)
    if _sha256(cases_path) != disagreement_input["cases_sha256"]:
        raise ValueError(f"held-out reliability cases SHA-256 mismatch: {model}")
    cases = _read(cases_path)
    if (
        cases.get("model") != model
        or cases.get("dataset") != "dtu-held-out"
        or int(cases.get("scene_count", -1)) != 22
        or int(cases.get("variant_count", -1)) != 4
        or int(cases.get("case_count", -1)) != 88
        or tuple(cases.get("variants", ())) != RELIABILITY_VARIANTS
        or len(cases.get("cases", [])) != 88
    ):
        raise ValueError(f"held-out reliability case design mismatch: {model}")

    disagreement_auroc = _auroc(disagreement)
    native_auroc = _auroc(native)
    point = disagreement_auroc["point_estimate"]
    native_point = native_auroc["point_estimate"]
    return {
        "model": model,
        "dataset": "dtu-held-out",
        "cases": str(cases_path),
        "cases_sha256": _sha256(cases_path),
        "failure_count": int(disagreement["failure_count"]),
        "case_count": 88,
        "disagreement_auroc": disagreement_auroc,
        "native_auroc": native_auroc,
        "auroc_threshold": auroc_threshold,
        "point_estimate_gate_pass": (
            point is not None and float(point) >= auroc_threshold
        ),
        "disagreement_exceeds_native_point_estimate": (
            point is not None
            and native_point is not None
            and float(point) > float(native_point)
        ),
        "source_reports": {
            "disagreement": {
                "path": str(disagreement_path),
                "sha256": _sha256(disagreement_path),
            },
            "native": {"path": str(native_path), "sha256": _sha256(native_path)},
        },
    }


def _audit_repair(
    model: str,
    path: Path,
    *,
    recovery_threshold: float,
    clean_threshold: float,
) -> dict[str, object]:
    report = _read(path)
    if (
        report.get("model") != model
        or report.get("dataset") != "dtu-held-out"
        or int(report.get("scene_count", -1)) != 22
        or report.get("corrupt_variant") != "asymmetric_crop_075"
        or report.get("repaired_variant") != "canonical_asymmetric_crop_075"
        or float(report.get("recovery_threshold", -1.0)) != recovery_threshold
        or float(report.get("clean_relative_threshold", -1.0)) != clean_threshold
    ):
        raise ValueError(f"held-out repair design mismatch: {model}")
    metric = report["by_metric"][ROTATION_REPAIR_METRIC]
    if metric.get("status") != "available" or int(metric["scene_count"]) != 22:
        raise ValueError(f"held-out rotation repair metric is incomplete: {model}")
    gate = metric["promotion_gate"]
    recovery = metric["gap_recovery"]
    clean = metric["scene_bootstrap"]["metrics"].get(
        "clean_relative_degradation"
    )
    return {
        "model": model,
        "dataset": "dtu-held-out",
        "rotation_gap_recovery": recovery,
        "clean_relative_degradation": clean,
        "point_estimate_recovery_pass": bool(gate["point_recovery_pass"]),
        "point_estimate_clean_cost_pass": bool(gate["point_clean_cost_pass"]),
        "point_estimate_gate_pass": bool(
            gate["point_recovery_pass"] and gate["point_clean_cost_pass"]
        ),
        "confidence_bound_recovery_pass": bool(gate["confidence_bound_recovery_pass"]),
        "confidence_bound_clean_cost_pass": bool(
            gate["confidence_bound_clean_cost_pass"]
        ),
        "source": str(path),
        "source_sha256": _sha256(path),
    }


def audit_final_claims(
    mechanism_path: Path,
    reliability_paths: list[tuple[str, Path, Path]],
    repair_paths: list[tuple[str, Path]],
    *,
    expected_models: list[str],
    detector_auroc_threshold: float = 0.75,
    repair_recovery_threshold: float = 0.30,
    clean_relative_threshold: float = 0.02,
) -> dict[str, object]:
    if len(set(expected_models)) != len(expected_models) or not expected_models:
        raise ValueError("expected models must be non-empty and unique")
    reliability_by_model = {
        model: (first, second) for model, first, second in reliability_paths
    }
    repair_by_model = {model: path for model, path in repair_paths}
    if (
        len(reliability_by_model) != len(reliability_paths)
        or set(reliability_by_model) != set(expected_models)
        or len(repair_by_model) != len(repair_paths)
        or set(repair_by_model) != set(expected_models)
    ):
        raise ValueError("final held-out model inputs do not match expected models")
    mechanism = _audit_mechanism(mechanism_path, expected_models=expected_models)
    reliability = [
        _audit_reliability_pair(
            model,
            *reliability_by_model[model],
            auroc_threshold=detector_auroc_threshold,
        )
        for model in expected_models
    ]
    repair = [
        _audit_repair(
            model,
            repair_by_model[model],
            recovery_threshold=repair_recovery_threshold,
            clean_threshold=clean_relative_threshold,
        )
        for model in expected_models
    ]
    detector_pass = all(record["point_estimate_gate_pass"] for record in reliability)
    detector_beats_native = all(
        record["disagreement_exceeds_native_point_estimate"] for record in reliability
    )
    repair_pass = all(record["point_estimate_gate_pass"] for record in repair)
    return {
        "schema_version": "1.0",
        "status": "complete",
        "evidence_complete": True,
        "models": expected_models,
        "mechanism": mechanism,
        "held_out_reliability": reliability,
        "held_out_repair": repair,
        "claim_gates": {
            "two_family_two_dataset_mechanism": mechanism[
                "meets_two_family_two_dataset_gate"
            ],
            "held_out_detector_all_models": detector_pass,
            "held_out_detector_exceeds_native_all_models": detector_beats_native,
            "cross_dataset_rotation_repair_all_models": repair_pass,
        },
        "interpretation": {
            "failed_gates_remain_negative_results": True,
            "generic_geometry_repair_not_implied": True,
            "review_score_not_computed_from_claim_gates": True,
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    report = audit_final_claims(
        args.mechanism,
        [
            (model, Path(disagreement), Path(native))
            for model, disagreement, native in args.reliability
        ],
        [(model, Path(path)) for model, path in args.repair],
        expected_models=args.expected_models,
        detector_auroc_threshold=args.detector_auroc_threshold,
        repair_recovery_threshold=args.repair_recovery_threshold,
        clean_relative_threshold=args.clean_relative_threshold,
    )
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "claim_gates": report["claim_gates"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
