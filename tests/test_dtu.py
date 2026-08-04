from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.spatial.transform import Rotation

from camcanon3r.dtu import (
    decompose_projection_matrix,
    evaluate_dtu_prediction,
    read_dtu_projection,
    read_ply_vertices,
)


def test_decompose_projection_matrix_recovers_camera_up_to_scale(
    tmp_path: Path,
) -> None:
    intrinsic = np.array([[1200.0, 3.0, 800.0], [0.0, 1180.0, 600.0], [0.0, 0.0, 1.0]])
    rotation = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
    translation = np.array([20.0, -30.0, 900.0])
    projection = -2.5 * intrinsic @ np.column_stack([rotation, translation])

    actual_intrinsic, actual_extrinsic = decompose_projection_matrix(projection)
    assert actual_intrinsic == pytest.approx(intrinsic)
    assert actual_extrinsic[:, :3] == pytest.approx(rotation)
    assert actual_extrinsic[:, 3] == pytest.approx(translation)

    path = tmp_path / "pos_023.txt"
    np.savetxt(path, projection)
    file_intrinsic, file_extrinsic = read_dtu_projection(path)
    assert file_intrinsic == pytest.approx(intrinsic)
    assert file_extrinsic == pytest.approx(actual_extrinsic)


def test_read_ply_vertices_supports_ascii_and_binary(tmp_path: Path) -> None:
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ascii_path = tmp_path / "ascii.ply"
    ascii_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "end_header\n"
        "1 2 3 20\n"
        "4 5 6 30\n",
        encoding="ascii",
    )
    assert read_ply_vertices(ascii_path) == pytest.approx(expected)

    binary_path = tmp_path / "binary.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "end_header\n"
    ).encode("ascii")
    with binary_path.open("wb") as handle:
        handle.write(header)
        for point, red in zip(expected, (20, 30), strict=True):
            handle.write(struct.pack("<fffB", *point, red))
    assert read_ply_vertices(binary_path) == pytest.approx(expected)


def test_evaluate_dtu_prediction_recovers_exact_camera_geometry(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    intrinsic = np.array([[1200.0, 0.0, 800.0], [0.0, 1180.0, 600.0], [0.0, 0.0, 1.0]])
    camera_ids = (23, 26, 29)
    extrinsics = []
    for index, camera_id in enumerate(camera_ids):
        rotation = Rotation.from_rotvec([0.0, 0.04 * index, 0.0]).as_matrix()
        translation = np.array([40.0 * index, 0.0, 900.0])
        extrinsic = np.column_stack([rotation, translation])
        extrinsics.append(extrinsic)
        np.savetxt(calibration / f"pos_{camera_id:03d}.txt", intrinsic @ extrinsic)
    affine = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
    prediction = tmp_path / "identity.npz"
    np.savez_compressed(
        prediction,
        extrinsic=np.stack(extrinsics),
        intrinsic=np.stack([affine @ intrinsic] * 3),
        source_to_model_affine=np.stack([affine] * 3),
        world_points=np.zeros((3, 2, 2, 3)),
    )
    inputs = [f"rect_{camera_id:03d}_3_r5000.png" for camera_id in camera_ids]
    prediction.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": inputs,
                "spatial_transforms": [
                    {"input": name, "input_size": [1600, 1200]} for name in inputs
                ],
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_dtu_prediction(prediction, calibration, scan=1, gt_root=None)
    assert result["relative_rotation_degrees"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert result["translation_direction_degrees"]["median"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert result["intrinsics"]["focal_relative_error"]["median"] == pytest.approx(
        0.0, abs=1e-10
    )
    assert result["intrinsics"]["principal_point_normalized_error"][
        "median"
    ] == pytest.approx(0.0, abs=1e-10)


def test_evaluate_dtu_prediction_reports_exact_synthetic_point_distances(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    gt_root = tmp_path / "gt"
    (gt_root / "Points/stl").mkdir(parents=True)
    (gt_root / "ObsMask").mkdir(parents=True)
    intrinsic = np.array([[1200.0, 0.0, 800.0], [0.0, 1180.0, 600.0], [0.0, 0.0, 1.0]])
    camera_ids = (23, 26, 29)
    camera_centers = (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    extrinsics = []
    for camera_id, center in zip(camera_ids, camera_centers, strict=True):
        extrinsic = np.column_stack([np.eye(3), -center])
        extrinsics.append(extrinsic)
        np.savetxt(calibration / f"pos_{camera_id:03d}.txt", intrinsic @ extrinsic)
    point_map = np.array(
        [
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            [[1.0, 2.0, 1.0], [2.0, 2.0, 1.0]],
        ]
    )
    world_points = np.stack([point_map] * 3)
    target_points = point_map.reshape(-1, 3)
    point_path = gt_root / "Points/stl/stl001_total.ply"
    point_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(target_points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n" + "".join(f"{x} {y} {z}\n" for x, y, z in target_points),
        encoding="ascii",
    )
    observation_mask = np.ones((5, 5, 5), dtype=np.uint8)
    # Official DTU applies ObsMask only to prediction-to-GT accuracy. The exact
    # predicted point at this masked target must remain available to the
    # GT-to-prediction completeness query.
    observation_mask[2, 2, 1] = 0
    savemat(
        gt_root / "ObsMask/ObsMask1_10.mat",
        {
            "ObsMask": observation_mask,
            "BB": np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]),
            "Res": np.array([[1.0]]),
        },
    )
    savemat(gt_root / "ObsMask/Plane1.mat", {"P": np.array([0.0, 0.0, 1.0, 0.0])})

    prediction = tmp_path / "identity.npz"
    np.savez_compressed(
        prediction,
        extrinsic=np.stack(extrinsics),
        intrinsic=np.stack([intrinsic] * 3),
        source_to_model_affine=np.stack([np.eye(3)] * 3),
        world_points=world_points,
    )
    inputs = [f"rect_{camera_id:03d}_3_r5000.png" for camera_id in camera_ids]
    prediction.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": inputs,
                "spatial_transforms": [
                    {"input": name, "input_size": [2, 2]} for name in inputs
                ],
            }
        ),
        encoding="utf-8",
    )
    result = evaluate_dtu_prediction(prediction, calibration, scan=1, gt_root=gt_root)
    point_cloud = result["point_cloud"]
    assert point_cloud["accuracy_millimeters"]["mean"] == pytest.approx(0.0)
    assert point_cloud["completeness_millimeters"]["mean"] == pytest.approx(0.0)
    assert point_cloud["alignment"]["scale"] == pytest.approx(1.0)
    assert point_cloud["predicted_points_evaluated"] == 4
    assert point_cloud["predicted_points_in_observation_mask"] == 3

    collapsed = np.repeat(extrinsics[0][None], 3, axis=0)
    np.savez_compressed(
        prediction,
        extrinsic=collapsed,
        intrinsic=np.stack([intrinsic] * 3),
        source_to_model_affine=np.stack([np.eye(3)] * 3),
        world_points=world_points,
    )
    undefined = evaluate_dtu_prediction(
        prediction, calibration, scan=1, gt_root=gt_root
    )["point_cloud"]
    assert undefined["status"] == ("undefined_degenerate_camera_center_alignment")
    assert undefined["accuracy_millimeters"]["mean"] is None
    assert undefined["predicted_source_supported_points_before_alignment"] == 12
