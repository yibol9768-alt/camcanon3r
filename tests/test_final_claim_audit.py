import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_final_claims import audit_final_claims


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    mechanism = tmp_path / "mechanism.json"
    _write(
        mechanism,
        {
            "analyses": {
                "vggt/eth3d": {
                    "dataset": "eth3d",
                    "analysis": {"scene_count": 13, "variant_count": 11},
                },
                "vggt/dtu": {
                    "dataset": "dtu",
                    "analysis": {"scene_count": 22, "variant_count": 11},
                },
            },
            "family_support": {"independent_asymmetric_crop": {}},
            "hypothesis_gate": {
                "required_family_count": 2,
                "required_dataset_count_per_family": 2,
                "evaluated_datasets": ["eth3d", "dtu"],
                "meets_two_family_two_dataset_gate": True,
            },
        },
    )
    cases = tmp_path / "cases.json"
    _write(
        cases,
        {
            "model": "vggt",
            "dataset": "dtu-held-out",
            "scene_count": 22,
            "variant_count": 4,
            "case_count": 88,
            "variants": [
                "identity",
                "center_crop_075",
                "asymmetric_crop_075",
                "letterbox_square",
            ],
            "cases": [{} for _ in range(88)],
        },
    )
    cases_sha = hashlib.sha256(cases.read_bytes()).hexdigest()

    def reliability(path: Path, field: str, auroc: float) -> None:
        _write(
            path,
            {
                "failure_definition": {"error_operator": ">", "threshold": 2.0},
                "case_count": 88,
                "scene_count": 22,
                "failure_count": 20,
                "bootstrap": {
                    "resampling_unit": "scene_cluster",
                    "replicates": 10000,
                    "seed": 17,
                    "confidence_level": 0.95,
                },
                "input": {
                    "cases": str(cases),
                    "cases_sha256": cases_sha,
                    "error_field": "ground_truth.rotation_median_degrees",
                    "uncertainty_field": field,
                },
                "auroc": {
                    "status": "defined",
                    "point_estimate": auroc,
                    "lower": auroc - 0.1,
                    "upper": min(1.0, auroc + 0.1),
                    "valid_replicates": 10000,
                    "undefined_replicates": 0,
                },
            },
        )

    disagreement = tmp_path / "disagreement.json"
    native = tmp_path / "native.json"
    reliability(disagreement, "scores.rotation_disagreement_degrees", 0.82)
    reliability(native, "scores.native_uncertainty", 0.61)
    repair = tmp_path / "repair.json"
    _write(
        repair,
        {
            "model": "vggt",
            "dataset": "dtu-held-out",
            "scene_count": 22,
            "corrupt_variant": "asymmetric_crop_075",
            "repaired_variant": "canonical_asymmetric_crop_075",
            "recovery_threshold": 0.30,
            "clean_relative_threshold": 0.02,
            "by_metric": {
                "relative_rotation_median_degrees": {
                    "status": "available",
                    "scene_count": 22,
                    "gap_recovery": {
                        "point_estimate": 0.5,
                        "lower": 0.2,
                        "upper": 0.8,
                    },
                    "scene_bootstrap": {
                        "metrics": {
                            "clean_relative_degradation": {
                                "point_estimate": 0.0,
                                "lower": 0.0,
                                "upper": 0.0,
                            }
                        }
                    },
                    "promotion_gate": {
                        "point_recovery_pass": True,
                        "point_clean_cost_pass": True,
                        "confidence_bound_recovery_pass": False,
                        "confidence_bound_clean_cost_pass": True,
                    },
                }
            },
        },
    )
    support_summaries = []
    for dataset, scene_count in (("eth3d", 13), ("dtu", 22)):
        path = tmp_path / f"support_{dataset}.json"
        _write(path, {"dataset": dataset, "scene_count": scene_count})
        support_summaries.append((dataset, scene_count, path))
    support_protocol = tmp_path / "support_protocol.json"
    _write(support_protocol, {"minimum_rotation_delta_degrees": 2.0})
    support = tmp_path / "support.json"
    _write(
        support,
        {
            "status": "complete",
            "models_are_separate": True,
            "datasets_are_separate": True,
            "protocol": str(support_protocol),
            "protocol_sha256": hashlib.sha256(
                support_protocol.read_bytes()
            ).hexdigest(),
            "analyses": [
                {
                    "model": "vggt",
                    "dataset": dataset,
                    "scene_count": scene_count,
                    "variants": [
                        "letterbox_square",
                        "shared_asymmetric_letterbox_square",
                        "asymmetric_letterbox_square",
                    ],
                    "source": str(path),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for dataset, scene_count, path in support_summaries
            ],
            "promotion_gate": {
                "variant": "asymmetric_letterbox_square",
                "rotation_delta_threshold_degrees": 2.0,
                "support": [
                    {
                        "model": "vggt",
                        "dataset": dataset,
                        "rotation_delta": {"point_estimate": 3.0},
                        "crosses_point_estimate_threshold": True,
                    }
                    for dataset, _, _ in support_summaries
                ],
                "passes_all_models_and_datasets": True,
            },
        },
    )
    return mechanism, disagreement, native, repair, support


def test_final_claim_audit_keeps_completeness_and_promotion_separate(
    tmp_path: Path,
) -> None:
    mechanism, disagreement, native, repair, support = _fixture(tmp_path)
    report = audit_final_claims(
        mechanism,
        [("vggt", disagreement, native)],
        [("vggt", repair)],
        support_control_path=support,
        expected_models=["vggt"],
    )
    assert report["evidence_complete"] is True
    assert report["claim_gates"] == {
        "two_family_two_dataset_mechanism": True,
        "support_preserving_coordinate_control_all_models_datasets": True,
        "held_out_detector_all_models": True,
        "held_out_detector_exceeds_native_all_models": True,
        "cross_dataset_rotation_repair_all_models": True,
    }
    assert report["held_out_repair"][0]["confidence_bound_recovery_pass"] is False


def test_final_claim_audit_rejects_tampered_support_gate(tmp_path: Path) -> None:
    mechanism, disagreement, native, repair, support = _fixture(tmp_path)
    payload = json.loads(support.read_text(encoding="utf-8"))
    payload["promotion_gate"]["support"][0]["rotation_delta"]["point_estimate"] = 1.0
    _write(support, payload)
    with pytest.raises(ValueError, match="promotion decision is inconsistent"):
        audit_final_claims(
            mechanism,
            [("vggt", disagreement, native)],
            [("vggt", repair)],
            support_control_path=support,
            expected_models=["vggt"],
        )


def test_final_claim_audit_retains_failed_detector_as_negative_result(
    tmp_path: Path,
) -> None:
    mechanism, disagreement, native, repair, support = _fixture(tmp_path)
    payload = json.loads(disagreement.read_text(encoding="utf-8"))
    payload["auroc"]["point_estimate"] = 0.70
    _write(disagreement, payload)
    report = audit_final_claims(
        mechanism,
        [("vggt", disagreement, native)],
        [("vggt", repair)],
        support_control_path=support,
        expected_models=["vggt"],
    )
    assert report["evidence_complete"] is True
    assert report["claim_gates"]["held_out_detector_all_models"] is False
