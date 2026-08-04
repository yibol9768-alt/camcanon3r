import json
from pathlib import Path

import pytest

from scripts.evaluate_dtu_selection import (
    EVALUATION_PROTOCOL_VERSION,
    _expected_jobs,
    _validate_existing,
)


def _fixture(root: Path) -> tuple[Path, Path, Path, Path, list[str], list[str]]:
    selection_root = root / "selection"
    prediction_root = root / "predictions"
    results_root = root / "results"
    protocol_path = root / "dtu_protocol.json"
    variant_config_path = root / "variants.json"
    variants = ["identity", "crop"]
    point_variants = ["identity"]
    variant_config_path.write_text(
        json.dumps(
            {
                "frozen_before_benchmark_scale_mechanism_results": True,
                "ordered_variants": variants,
            }
        ),
        encoding="utf-8",
    )
    protocol_path.write_text(
        json.dumps(
            {
                "frozen_before_dtu_gt_inspection": True,
                "evaluation_scans": list(range(1, 23)),
                "rectified_archive_camera_ids_one_based": [23, 26, 29],
                "lighting_index": 3,
                "variant_config": str(variant_config_path),
                "confirmatory_variants": point_variants,
            }
        ),
        encoding="utf-8",
    )
    calibration = selection_root / "calibration/cal18"
    calibration.mkdir(parents=True)
    for camera_id in (23, 26, 29):
        (calibration / f"pos_{camera_id:03d}.txt").touch()
    inputs = [f"rect_{camera_id:03d}_3_r5000.png" for camera_id in (23, 26, 29)]
    for scan in range(1, 23):
        source = selection_root / f"source/scan{scan}"
        source.mkdir(parents=True)
        for name in inputs:
            (source / name).touch()
        prediction_scene = prediction_root / f"scan{scan}"
        prediction_scene.mkdir(parents=True)
        for variant in variants:
            (prediction_scene / f"{variant}.npz").touch()
            (prediction_scene / f"{variant}.json").write_text(
                json.dumps({"inputs": inputs}), encoding="utf-8"
            )
        gt_root = selection_root / "gt"
        (gt_root / "Points/stl").mkdir(parents=True, exist_ok=True)
        (gt_root / "ObsMask").mkdir(parents=True, exist_ok=True)
        (gt_root / f"Points/stl/stl{scan:03d}_total.ply").touch()
        (gt_root / f"ObsMask/ObsMask{scan}_10.mat").touch()
        (gt_root / f"ObsMask/Plane{scan}.mat").touch()
    return (
        selection_root,
        prediction_root,
        results_root,
        protocol_path,
        variants,
        point_variants,
    )


def test_dtu_job_plan_is_exactly_frozen(tmp_path: Path) -> None:
    selection, predictions, results, protocol, variants, point_variants = _fixture(
        tmp_path
    )
    jobs, design = _expected_jobs(
        selection,
        predictions,
        results,
        protocol,
        variants=variants,
        point_variants=point_variants,
    )
    assert len(jobs) == 44
    assert (jobs[0]["scene"], jobs[0]["variant"]) == ("scan1", "identity")
    assert jobs[0]["point_metrics_requested"] is True
    assert jobs[1]["point_metrics_requested"] is False
    assert design["scans"] == list(range(1, 23))

    record = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "scene": jobs[0]["scene"],
        "scan": jobs[0]["scan"],
        "variant": jobs[0]["variant"],
        "prediction": str(Path(jobs[0]["prediction"]).resolve()),
        "calibration_dir": str(Path(jobs[0]["calibration_dir"]).resolve()),
        "gt_root": str(Path(jobs[0]["gt_root"]).resolve()),
        "point_metrics_requested": True,
        "protocol_sha256": jobs[0]["protocol_sha256"],
        "variant_config_sha256": jobs[0]["variant_config_sha256"],
    }
    _validate_existing(record, jobs[0])


def test_dtu_job_plan_rejects_variant_or_view_drift(tmp_path: Path) -> None:
    selection, predictions, results, protocol, variants, point_variants = _fixture(
        tmp_path
    )
    with pytest.raises(ValueError, match="frozen ordered design"):
        _expected_jobs(
            selection,
            predictions,
            results,
            protocol,
            variants=list(reversed(variants)),
            point_variants=point_variants,
        )
    metadata = predictions / "scan1/identity.json"
    metadata.write_text(
        json.dumps(
            {
                "inputs": [
                    "rect_026_3_r5000.png",
                    "rect_023_3_r5000.png",
                    "rect_029_3_r5000.png",
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="do not match frozen DTU views"):
        _expected_jobs(
            selection,
            predictions,
            results,
            protocol,
            variants=variants,
            point_variants=point_variants,
        )


def test_dtu_resume_rejects_stale_protocol_hash(tmp_path: Path) -> None:
    selection, predictions, results, protocol, variants, point_variants = _fixture(
        tmp_path
    )
    jobs, _ = _expected_jobs(
        selection,
        predictions,
        results,
        protocol,
        variants=variants,
        point_variants=point_variants,
    )
    job = jobs[0]
    with pytest.raises(ValueError, match="does not match frozen job"):
        _validate_existing(
            {
                "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
                "scene": job["scene"],
                "scan": job["scan"],
                "variant": job["variant"],
                "prediction": str(Path(job["prediction"]).resolve()),
                "calibration_dir": str(Path(job["calibration_dir"]).resolve()),
                "gt_root": str(Path(job["gt_root"]).resolve()),
                "point_metrics_requested": True,
                "protocol_sha256": "stale",
                "variant_config_sha256": job["variant_config_sha256"],
            },
            job,
        )
