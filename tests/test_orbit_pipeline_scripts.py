import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from camcanon3r.metrics import pairwise_relative_pose_errors

MEMBER_BIASES = {
    "center": 0.0,
    "left": 2.5,
    "right": 5.5,
    "top": 3.0,
    "bottom": 1.0,
    "top_left": 5.5,
    "bottom_right": 6.5,
    "top_right": 8.5,
    "bottom_left": 3.5,
}


def _two_view(angle_degrees: float) -> np.ndarray:
    rotations = np.stack(
        [
            np.eye(3),
            Rotation.from_euler("z", angle_degrees, degrees=True).as_matrix(),
        ]
    )
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1]])
    translations = -np.einsum("vij,vj->vi", rotations, centers)
    return np.concatenate([rotations, translations[:, :, None]], axis=2)


def _write_prediction(
    path: Path,
    extrinsics: np.ndarray,
    *,
    confidence: float,
    inputs: tuple[str, str] = ("image1.png", "image2.png"),
    prepared_dir: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "extrinsic": extrinsics,
        "world_points_conf": np.asarray([confidence]),
    }
    if prepared_dir is not None:
        view_count = len(inputs)
        grid_y, grid_x = np.mgrid[:5, :5]
        world_points = np.stack(
            [
                np.stack([grid_x, grid_y, np.full((5, 5), 2.0 + view)], axis=-1)
                for view in range(view_count)
            ]
        )
        identity = np.repeat(np.eye(3)[None], view_count, axis=0)
        arrays.update(
            {
                "intrinsic": identity,
                "depth": world_points[..., 2],
                "depth_conf": np.full((view_count, 5, 5), confidence + 1.0),
                "world_points": world_points,
                "world_points_conf": np.full((view_count, 5, 5), confidence + 1.0),
                "model_preprocess_affine": identity,
                "protocol_affine": identity,
                "source_to_model_affine": identity,
            }
        )
    np.savez_compressed(path, **arrays)
    metadata = {"inputs": list(inputs)}
    if prepared_dir is not None:
        metadata["scene_directory"] = str(prepared_dir.resolve())
    path.with_suffix(".json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )


def _write_orbit_predictions(root: Path, scene: str) -> None:
    prepared_scene = root.parent / "prepared" / scene
    support_paths = []
    for name in ("image1.png", "image2.png"):
        support_path = prepared_scene / "source_masks" / name
        support_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ones((5, 5), dtype=np.uint8) * 255).save(support_path)
        support_paths.append(support_path)
    variants = []
    for index, (label, bias) in enumerate(MEMBER_BIASES.items()):
        prepared_dir = prepared_scene / f"orbit_{label}"
        variants.append(
            {
                "name": f"orbit_{label}",
                "images": [
                    {
                        "output": f"orbit_{label}/{name}",
                        "orbit": {"source_mask": str(mask.resolve())},
                    }
                    for name, mask in zip(
                        ("image1.png", "image2.png"), support_paths, strict=True
                    )
                ],
            }
        )
        _write_prediction(
            root / scene / f"orbit_{label}.npz",
            _two_view(bias),
            confidence=float(index),
            prepared_dir=prepared_dir,
        )
    prepared_scene.mkdir(parents=True, exist_ok=True)
    (prepared_scene / "manifest.json").write_text(
        json.dumps({"variants": variants}) + "\n", encoding="utf-8"
    )


def test_project_orbit_sweep_is_resumable_and_fuses_geometry(tmp_path):
    predictions = tmp_path / "predictions"
    projections = tmp_path / "projections"
    report = tmp_path / "projection_report.json"
    _write_orbit_predictions(predictions, "scene1")

    arguments = (
        "scripts/project_orbit_sweep.py",
        str(predictions),
        str(projections),
        "--protocol",
        "configs/orbit_projection_protocol.json",
        "--scenes",
        "scene1",
        "--report",
        str(report),
    )
    _run(*arguments)
    resumed = _run(*arguments, "--resume")

    payload = json.loads(report.read_text(encoding="utf-8"))
    metadata = json.loads(
        (projections / "scene1/response_projection.json").read_text(encoding="utf-8")
    )
    with np.load(projections / "scene1/response_projection.npz") as prediction:
        response = prediction["extrinsic"]
        assert set(prediction.files) == {"extrinsic", "rotation", "camera_center"}
    fusion_metadata = json.loads(
        (projections / "scene1/response_fusion.json").read_text(encoding="utf-8")
    )
    with np.load(projections / "scene1/response_fusion.npz") as prediction:
        assert {
            "extrinsic",
            "intrinsic",
            "depth",
            "world_points",
            "source_to_model_affine",
        } <= set(prediction.files)
    error = np.median(
        pairwise_relative_pose_errors(_two_view(0.0), response)["rotation_degrees"]
    )
    assert payload["status"] == "complete"
    assert payload["record_count"] == 1
    assert metadata["camera_only"] is True
    assert fusion_metadata["camera_only"] is False
    assert fusion_metadata["diagnostics"]["common_source_support_only"] is True
    assert metadata["diagnostics"]["ground_truth_used"] is False
    assert error < 0.5
    assert '"status": "complete"' in resumed.stdout


def _write_eth3d_calibration(root: Path) -> None:
    calibration = root / "scene1/dslr_calibration_jpg"
    calibration.mkdir(parents=True)
    (calibration / "cameras.txt").write_text(
        "# synthetic COLMAP cameras\n1 PINHOLE 5 5 4 4 2 2\n",
        encoding="utf-8",
    )
    (calibration / "images.txt").write_text(
        "# synthetic COLMAP images\n"
        "1 1 0 0 0 0 0 0 1 image1.JPG\n"
        "0 0 -1\n"
        "2 1 0 0 0 -1 0 0 1 image2.JPG\n"
        "0 0 -1\n",
        encoding="utf-8",
    )
    depth = root / "scene1/ground_truth_depth/dslr_images"
    depth.mkdir(parents=True)
    np.full((5, 5), 2.0, dtype="<f4").tofile(depth / "image1.JPG")
    np.full((5, 5), 3.0, dtype="<f4").tofile(depth / "image2.JPG")


def test_evaluate_orbit_projection_keeps_oracle_separate(tmp_path):
    selection = tmp_path / "selection"
    predictions = tmp_path / "predictions"
    projections = tmp_path / "projections"
    identity = tmp_path / "identity"
    analytic = tmp_path / "analytic"
    projection_report = tmp_path / "projection_report.json"
    evaluation = tmp_path / "evaluation.json"
    fusion_results = tmp_path / "fusion_results"
    fusion_summary = tmp_path / "fusion_summary.json"
    _write_eth3d_calibration(selection)
    _write_orbit_predictions(predictions, "scene1")
    _write_prediction(
        identity / "scene1/identity.npz",
        _two_view(0.0),
        confidence=1.0,
        prepared_dir=tmp_path / "identity_prepared",
    )
    _write_prediction(
        analytic / "scene1/canonical_asymmetric_crop_075.npz",
        _two_view(3.0),
        confidence=1.0,
        prepared_dir=tmp_path / "analytic_prepared",
    )
    _run(
        "scripts/project_orbit_sweep.py",
        str(predictions),
        str(projections),
        "--protocol",
        "configs/orbit_projection_protocol.json",
        "--scenes",
        "scene1",
        "--report",
        str(projection_report),
    )

    completed = _run(
        "scripts/evaluate_orbit_projection.py",
        str(selection),
        str(predictions),
        str(projections),
        str(identity),
        str(analytic),
        str(evaluation),
        "--protocol",
        "configs/orbit_projection_protocol.json",
        "--dataset",
        "eth3d",
        "--dataset-label",
        "synthetic-eth3d",
        "--model",
        "vggt",
        "--scenes",
        "scene1",
    )
    fusion_completed = _run(
        "scripts/evaluate_orbit_fusion.py",
        str(selection),
        str(projections),
        str(identity),
        str(analytic),
        str(fusion_results),
        str(fusion_summary),
        "--protocol",
        "configs/orbit_projection_protocol.json",
        "--dataset",
        "eth3d",
        "--dataset-label",
        "synthetic-eth3d",
        "--model",
        "vggt",
        "--scenes",
        "scene1",
    )

    report = json.loads(evaluation.read_text(encoding="utf-8"))
    provenance = report["per_scene"]["scene1"]["provenance"]
    assert report["status"] == "complete"
    assert report["method_ground_truth_used"] is False
    assert report["oracle_ground_truth_used"] is True
    assert provenance["ground_truth_used_by_oracle_only"] is True
    assert provenance["oracle_label"] in MEMBER_BIASES
    assert report["summary"]["promotion"]["promotion_pass"] is True
    assert '"promotion_pass": true' in completed.stdout
    full_report = json.loads(fusion_summary.read_text(encoding="utf-8"))
    assert full_report["status"] == "complete"
    assert full_report["method_ground_truth_used"] is False
    assert full_report["evaluation_count"] == 3
    assert full_report["summary"]["reference_variant"] == "analytic_repair"
    assert '"status": "complete"' in fusion_completed.stdout


def test_projection_resume_rejects_changed_source_prediction(tmp_path):
    predictions = tmp_path / "predictions"
    projections = tmp_path / "projections"
    report = tmp_path / "projection_report.json"
    _write_orbit_predictions(predictions, "scene1")
    arguments = (
        "scripts/project_orbit_sweep.py",
        str(predictions),
        str(projections),
        "--protocol",
        "configs/orbit_projection_protocol.json",
        "--scenes",
        "scene1",
        "--report",
        str(report),
    )
    _run(*arguments)
    _write_prediction(
        predictions / "scene1/orbit_center.npz",
        _two_view(0.4),
        confidence=0.0,
    )

    with pytest.raises(subprocess.CalledProcessError):
        _run(*arguments, "--resume")
