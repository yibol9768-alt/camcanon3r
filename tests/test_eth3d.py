from pathlib import Path

import numpy as np

from camcanon3r.eth3d import (
    ColmapCamera,
    colmap_camera_matrix,
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
    (calibration / "cameras.txt").write_text("0 PINHOLE 3 2 10 10 1 1\n")
    (calibration / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 0 images/a.JPG\n\n"
        "2 1 0 0 0 1 0 0 0 images/b.JPG\n\n"
    )
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    np.full((2, 3), 2.0, dtype="<f4").tofile(depth_dir / "a.JPG")
    np.full((2, 3), 2.0, dtype="<f4").tofile(depth_dir / "b.JPG")

    prediction = tmp_path / "prediction.npz"
    extrinsic = np.stack(
        [
            np.column_stack([np.eye(3), np.zeros(3)]),
            np.column_stack([np.eye(3), np.array([1.0, 0.0, 0.0])]),
        ]
    )
    np.savez(
        prediction,
        extrinsic=extrinsic,
        intrinsic=np.repeat(
            np.asarray([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])[
                None
            ],
            2,
            axis=0,
        ),
        depth=np.full((2, 2, 3), 1.0),
        source_to_model_affine=np.repeat(np.eye(3)[None], 2, axis=0),
    )
    prediction.with_suffix(".json").write_text(
        '{"inputs": ["a.png", "b.png"]}\n'
    )

    pose_only = evaluate_eth3d_prediction(
        prediction, calibration, depth_dir=None
    )
    with_depth = evaluate_eth3d_prediction(
        prediction, calibration, depth_dir=depth_dir
    )
    assert pose_only["depth"] is None
    assert pose_only["variant"] == "prediction"
    assert pose_only["relative_rotation_degrees"]["median"] == 0.0
    assert pose_only["intrinsics"]["focal_relative_error"]["median"] == 0.0
    assert (
        pose_only["intrinsics"]["principal_point_normalized_error"]["median"]
        == 0.0
    )
    assert with_depth["depth"]["mean_abs_rel"] == 0.0
    assert with_depth["depth"]["scale"] == 2.0
