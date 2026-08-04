import json
from pathlib import Path

import pytest

from scripts.evaluate_dtu_repair_selection import (
    EVALUATION_PROTOCOL_VERSION,
    _expected_jobs,
    _validate_existing,
)


def _fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path, list[str], list[str]]:
    selection_root = root / "selection"
    repaired_root = root / "prepared_repair"
    prediction_root = root / "predictions"
    results_root = root / "results"
    base_protocol_path = root / "dtu_protocol.json"
    repair_protocol_path = root / "dtu_repair_protocol.json"
    qualitative_protocol_path = root / "qualitative_protocol.json"
    preparation_audit_path = root / "preparation_audit.json"
    source_variants = ["identity", "asymmetric_crop_075"]
    variants = ["identity", "canonical_asymmetric_crop_075"]

    scans = list(range(1, 23))
    scenes = [f"scan{scan}" for scan in scans]
    base_protocol_path.write_text(
        json.dumps(
            {
                "frozen_before_dtu_gt_inspection": True,
                "evaluation_scans": scans,
                "rectified_archive_camera_ids_one_based": [23, 26, 29],
                "lighting_index": 3,
            }
        ),
        encoding="utf-8",
    )
    qualitative_protocol_path.write_text(
        json.dumps(
            {
                "variants": [
                    "identity",
                    "asymmetric_crop_075",
                    "canonical_asymmetric_crop_075",
                ]
            }
        ),
        encoding="utf-8",
    )
    repair_protocol_path.write_text(
        json.dumps(
            {
                "frozen_before_dtu_gt_inspection": True,
                "base_protocol": str(base_protocol_path),
                "qualitative_protocol": str(qualitative_protocol_path),
                "source_variants": source_variants,
                "ordered_repaired_variants": variants,
                "point_metrics_variants": variants,
                "fill_policy": "neutral_gray",
            }
        ),
        encoding="utf-8",
    )
    repaired_root.mkdir()
    preparation_audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "scene_count": 22,
                "scenes": scenes,
                "source_variants": source_variants,
                "fill_policy": "neutral_gray",
                "output_variants": variants,
                "image_count": 132,
                "mask_count": 132,
                "identity_pixel_matches": 66,
                "tree_sha256": "a" * 64,
                "repaired_root": str(repaired_root.resolve()),
            }
        ),
        encoding="utf-8",
    )

    calibration = selection_root / "calibration/cal18"
    calibration.mkdir(parents=True)
    for camera_id in (23, 26, 29):
        (calibration / f"pos_{camera_id:03d}.txt").touch()
    inputs = [f"rect_{camera_id:03d}_3_r5000.png" for camera_id in (23, 26, 29)]
    gt_root = selection_root / "gt"
    (gt_root / "Points/stl").mkdir(parents=True)
    (gt_root / "ObsMask").mkdir(parents=True)
    for scan, scene in zip(scans, scenes, strict=True):
        source = selection_root / "source" / scene
        source.mkdir(parents=True)
        for name in inputs:
            (source / name).touch()
        prediction_scene = prediction_root / scene
        prediction_scene.mkdir(parents=True)
        for variant in variants:
            (prediction_scene / f"{variant}.npz").touch()
            (prediction_scene / f"{variant}.json").write_text(
                json.dumps(
                    {
                        "inputs": inputs,
                        "scene_directory": str(
                            (repaired_root / scene / variant).resolve()
                        ),
                    }
                ),
                encoding="utf-8",
            )
        (gt_root / f"Points/stl/stl{scan:03d}_total.ply").touch()
        (gt_root / f"ObsMask/ObsMask{scan}_10.mat").touch()
        (gt_root / f"ObsMask/Plane{scan}.mat").touch()
    return (
        selection_root,
        prediction_root,
        results_root,
        repair_protocol_path,
        preparation_audit_path,
        variants,
        variants,
    )


def test_dtu_repair_job_plan_is_exact_and_provenance_bound(tmp_path: Path) -> None:
    selection, predictions, results, protocol, audit, variants, point_variants = (
        _fixture(tmp_path)
    )
    jobs, design = _expected_jobs(
        selection,
        predictions,
        results,
        protocol,
        audit,
        variants=variants,
        point_variants=point_variants,
    )
    assert len(jobs) == 44
    assert jobs[0]["scene"] == "scan1"
    assert jobs[0]["variant"] == "identity"
    assert jobs[0]["point_metrics_requested"] is True
    assert design["preparation_tree_sha256"] == "a" * 64

    job = jobs[0]
    record = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "scene": job["scene"],
        "scan": job["scan"],
        "variant": job["variant"],
        "prediction": str(Path(job["prediction"]).resolve()),
        "calibration_dir": str(Path(job["calibration_dir"]).resolve()),
        "gt_root": str(Path(job["gt_root"]).resolve()),
        "point_metrics_requested": True,
        "protocol_sha256": job["protocol_sha256"],
        "base_protocol_sha256": job["base_protocol_sha256"],
        "qualitative_protocol_sha256": job["qualitative_protocol_sha256"],
        "preparation_audit_sha256": job["preparation_audit_sha256"],
        "preparation_tree_sha256": job["preparation_tree_sha256"],
    }
    _validate_existing(record, job)


def test_dtu_repair_job_plan_rejects_variant_and_audit_drift(tmp_path: Path) -> None:
    selection, predictions, results, protocol, audit, variants, point_variants = (
        _fixture(tmp_path)
    )
    with pytest.raises(ValueError, match="frozen ordered design"):
        _expected_jobs(
            selection,
            predictions,
            results,
            protocol,
            audit,
            variants=list(reversed(variants)),
            point_variants=point_variants,
        )

    report = json.loads(audit.read_text(encoding="utf-8"))
    report["fill_policy"] = "black"
    audit.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="preparation audit"):
        _expected_jobs(
            selection,
            predictions,
            results,
            protocol,
            audit,
            variants=variants,
            point_variants=point_variants,
        )


def test_dtu_repair_job_plan_rejects_prediction_tree_drift(tmp_path: Path) -> None:
    selection, predictions, results, protocol, audit, variants, point_variants = (
        _fixture(tmp_path)
    )
    metadata = predictions / "scan1/identity.json"
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record["scene_directory"] = "/wrong/tree"
    metadata.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited input tree"):
        _expected_jobs(
            selection,
            predictions,
            results,
            protocol,
            audit,
            variants=variants,
            point_variants=point_variants,
        )
