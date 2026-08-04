from pathlib import Path

import numpy as np

from camcanon3r.eth3d import (
    ColmapCamera,
    colmap_camera_matrix,
    colmap_unproject_pixels,
    evaluate_eth3d_prediction,
    open_eth3d_depth,
    quaternion_to_rotation,
    read_colmap_cameras,
    read_colmap_images,
)
from camcanon3r.metrics import aligned_depth_to_source_ground_truth


def test_eth3d_colmap_readers_and_depth_shape(tmp_path: Path) -> None:
    cameras = tmp_path / "cameras.txt"
    cameras.write_text("0 PINHOLE 3 2 10 10 1 1\n")
    images = tmp_path / "images.txt"
    images.write_text("1 1 0 0 0 1 2 3 0 images/a.JPG\n\n")
    depth_path = tmp_path / "a.JPG"
    np.arange(6, dtype="<f4").tofile(depth_path)
    camera = read_colmap_cameras(cameras)[0]
    image = read_colmap_images(images)["a"]
    depth = open_eth3d_depth(depth_path, width=camera.width, height=camera.height)
    np.testing.assert_allclose(image.extrinsic[:, :3], np.eye(3))
    np.testing.assert_allclose(image.extrinsic[:, 3], [1, 2, 3])
    np.testing.assert_allclose(depth, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_allclose(quaternion_to_rotation((2, 0, 0, 0)), np.eye(3))


def test_colmap_camera_matrix_supports_eth3d_models() -> None:
    pinhole = ColmapCamera(0, "PINHOLE", 100, 80, (90, 91, 50, 40))
    raw = ColmapCamera(
        0,
        "THIN_PRISM_FISHEYE",
        100,
        80,
        (90, 91, 50, 40, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    expected = np.asarray([[90, 0, 50], [0, 91, 40], [0, 0, 1]])
    np.testing.assert_allclose(colmap_camera_matrix(pinhole), expected)
    np.testing.assert_allclose(colmap_camera_matrix(raw), expected)


def test_colmap_unprojection_matches_fisheye_then_thin_prism_order() -> None:
    zero_distortion = ColmapCamera(
        0,
        "THIN_PRISM_FISHEYE",
        100,
        80,
        (100, 100, 50, 40, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    normalized, valid = colmap_unproject_pixels(
        zero_distortion, np.asarray([[60.0, 40.0]])
    )
    assert valid.tolist() == [True]
    np.testing.assert_allclose(normalized, [[np.tan(0.1), 0.0]], atol=1e-12)

    parameters = (
        100,
        101,
        50,
        40,
        0.03,
        -0.01,
        0.002,
        -0.003,
        0.004,
        0.001,
        0.005,
        -0.004,
    )
    camera = ColmapCamera(0, "THIN_PRISM_FISHEYE", 100, 80, parameters)
    expected_normalized = np.asarray([[0.3, -0.2]])
    radius = np.linalg.norm(expected_normalized, axis=1)
    fisheye = expected_normalized * (np.arctan(radius) / radius)[:, None]
    x, y = fisheye[0]
    radius_squared = x * x + y * y
    k1, k2, p1, p2, k3, k4, sx1, sy1 = parameters[4:]
    radial = 1.0 + radius_squared * (
        k1 + radius_squared * (k2 + radius_squared * (k3 + radius_squared * k4))
    )
    distorted = np.asarray(
        [
            x * radial
            + 2.0 * p1 * x * y
            + p2 * (radius_squared + 2.0 * x * x)
            + sx1 * radius_squared,
            y * radial
            + 2.0 * p2 * x * y
            + p1 * (radius_squared + 2.0 * y * y)
            + sy1 * radius_squared,
        ]
    )
    pixels = distorted * parameters[:2] + parameters[2:4]
    normalized, valid = colmap_unproject_pixels(camera, pixels[None])
    assert valid.tolist() == [True]
    np.testing.assert_allclose(normalized, expected_normalized, atol=1e-9)


def test_depth_to_source_ground_truth_recovers_scale() -> None:
    prediction = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    ground_truth = [prediction[0] * 3.0]
    affine = np.eye(3)[None]
    result = aligned_depth_to_source_ground_truth(prediction, affine, ground_truth)
    assert result["scale"] == 3.0
    assert result["valid_pixels"] == 4
    assert result["mean_abs_rel"] == 0.0


def test_eth3d_prediction_supports_pose_only_and_raw_depth(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "cameras.txt").write_text(
        "0 PINHOLE 3 2 10 10 1 1\n"
        "1 PINHOLE 3 2 12 12 1 1\n"
    )
    (calibration / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 0 images/a.JPG\n\n"
        "2 1 0 0 0 1 0 0 0 images/b.JPG\n\n"
        "3 1 0 0 0 0 1 0 1 images/c.JPG\n\n"
    )
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    np.full((2, 3), 2.0, dtype="<f4").tofile(depth_dir / "a.JPG")
    np.full((2, 3), 2.0, dtype="<f4").tofile(depth_dir / "b.JPG")
    np.full((2, 3), 2.0, dtype="<f4").tofile(depth_dir / "c.JPG")

    prediction = tmp_path / "prediction.npz"
    extrinsic = np.stack(
        [
            np.column_stack([np.eye(3), np.zeros(3)]),
            np.column_stack([np.eye(3), np.array([0.5, 0.0, 0.0])]),
            np.column_stack([np.eye(3), np.array([0.0, 0.5, 0.0])]),
        ]
    )
    grid_y, grid_x = np.mgrid[:2, :3]
    camera_points = [
        np.stack(
            [
                (grid_x - 1.0) / focal,
                (grid_y - 1.0) / focal,
                np.ones((2, 3)),
            ],
            axis=-1,
        )
        for focal in (10.0, 10.0, 12.0)
    ]
    world_points = np.stack(
        [
            points - item[:, 3]
            for points, item in zip(camera_points, extrinsic, strict=True)
        ]
    )
    predicted_intrinsics = np.stack(
        [
            [[focal, 0.0, 1.0], [0.0, focal, 1.0], [0.0, 0.0, 1.0]]
            for focal in (10.0, 10.0, 12.0)
        ]
    )
    np.savez(
        prediction,
        extrinsic=extrinsic,
        intrinsic=predicted_intrinsics,
        depth=np.full((3, 2, 3), 1.0),
        world_points=world_points,
        source_to_model_affine=np.repeat(np.eye(3)[None], 3, axis=0),
    )
    prediction.with_suffix(".json").write_text(
        '{"inputs": ["a.png", "b.png", "c.png"]}\n'
    )

    pose_only = evaluate_eth3d_prediction(
        prediction, calibration, depth_dir=None
    )
    with_depth = evaluate_eth3d_prediction(
        prediction, calibration, depth_dir=depth_dir
    )
    assert pose_only["depth"] is None
    assert pose_only["camera_ids"] == [0, 0, 1]
    assert pose_only["variant"] == "prediction"
    assert pose_only["relative_rotation_degrees"]["median"] == 0.0
    assert pose_only["intrinsics"]["focal_relative_error"]["median"] == 0.0
    assert (
        pose_only["intrinsics"]["principal_point_normalized_error"]["median"]
        == 0.0
    )
    assert with_depth["depth"]["mean_abs_rel"] == 0.0
    assert with_depth["depth"]["scale"] == 2.0
    assert with_depth["point_cloud"]["protocol"] == "raw_depth_backprojection"
    assert with_depth["point_cloud"]["accuracy_meters"]["mean"] < 1e-12
    assert with_depth["point_cloud"]["completeness_meters"]["mean"] < 1e-12

    np.savez(
        prediction,
        extrinsic=np.repeat(extrinsic[:1], 3, axis=0),
        intrinsic=predicted_intrinsics,
        depth=np.full((3, 2, 3), 1.0),
        world_points=world_points,
        source_to_model_affine=np.repeat(np.eye(3)[None], 3, axis=0),
    )
    collapsed = evaluate_eth3d_prediction(
        prediction, calibration, depth_dir=depth_dir
    )
    assert collapsed["depth"]["mean_abs_rel"] == 0.0
    assert (
        collapsed["point_cloud"]["status"]
        == "undefined_degenerate_camera_center_alignment"
    )
    assert collapsed["point_cloud"]["accuracy_meters"]["mean"] is None
