import json
from pathlib import Path

import pytest

from scripts.evaluate_eth3d_selection import _expected_jobs, _validate_existing


def _selection_fixture(root: Path) -> tuple[Path, Path, Path]:
    selection_root = root / "selection"
    prediction_root = root / "predictions"
    results_root = root / "results"
    scene = "office"
    (selection_root / scene / "dslr_calibration_jpg").mkdir(parents=True)
    (selection_root / scene / "ground_truth_depth/dslr_images").mkdir(
        parents=True
    )
    (selection_root / "selection_report.json").write_text(
        json.dumps(
            {
                "completed_at": "2026-08-04T00:00:00+00:00",
                "selection": {
                    "scenes": [
                        {
                            "scene": scene,
                            "image_names": ["DSC_0001.JPG", "DSC_0002.JPG"],
                        }
                    ]
                },
            }
        )
    )
    scene_predictions = prediction_root / scene
    scene_predictions.mkdir(parents=True)
    for variant in ("identity", "asymmetric_crop_075"):
        (scene_predictions / f"{variant}.npz").touch()
        (scene_predictions / f"{variant}.json").write_text(
            json.dumps({"inputs": ["DSC_0001.png", "DSC_0002.png"]})
        )
    return selection_root, prediction_root, results_root


def test_expected_eth3d_jobs_are_frozen_by_scene_variant_and_domain(
    tmp_path: Path,
) -> None:
    selection_root, prediction_root, results_root = _selection_fixture(tmp_path)
    jobs = _expected_jobs(
        selection_root,
        prediction_root,
        results_root,
        domain="raw",
        variants=["identity", "asymmetric_crop_075"],
    )
    assert [(job["scene"], job["variant"]) for job in jobs] == [
        ("office", "identity"),
        ("office", "asymmetric_crop_075"),
    ]
    assert jobs[0]["depth_dir"] == (
        selection_root / "office/ground_truth_depth/dslr_images"
    )
    record = {
        key: (
            str(Path(jobs[0][key]).resolve())
            if key in {"prediction", "calibration_dir"}
            else (
                str(Path(jobs[0][key]).resolve())
                if key == "depth_dir" and jobs[0][key]
                else jobs[0][key]
            )
        )
        for key in (
            "scene",
            "variant",
            "domain",
            "prediction",
            "calibration_dir",
            "depth_dir",
        )
    }
    _validate_existing(record, jobs[0])


def test_eth3d_job_plan_rejects_prediction_view_drift(tmp_path: Path) -> None:
    selection_root, prediction_root, results_root = _selection_fixture(tmp_path)
    (prediction_root / "office/identity.json").write_text(
        json.dumps({"inputs": ["DSC_0002.png", "DSC_0001.png"]})
    )
    with pytest.raises(RuntimeError, match="do not match frozen selection"):
        _expected_jobs(
            selection_root,
            prediction_root,
            results_root,
            domain="raw",
            variants=["identity"],
        )


def test_eth3d_resume_rejects_stale_evaluation_record(tmp_path: Path) -> None:
    selection_root, prediction_root, results_root = _selection_fixture(tmp_path)
    job = _expected_jobs(
        selection_root,
        prediction_root,
        results_root,
        domain="raw",
        variants=["identity"],
    )[0]
    with pytest.raises(ValueError, match="does not match frozen job"):
        _validate_existing(
            {
                "scene": "office",
                "variant": "identity",
                "domain": "raw",
                "prediction": "/stale/prediction.npz",
                "calibration_dir": str(Path(job["calibration_dir"]).resolve()),
                "depth_dir": str(Path(job["depth_dir"]).resolve()),
            },
            job,
        )
