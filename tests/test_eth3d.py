from pathlib import Path

import numpy as np

from camcanon3r.eth3d import (
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


def test_depth_to_source_ground_truth_recovers_scale() -> None:
    prediction = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    ground_truth = [prediction[0] * 3.0]
    affine = np.eye(3)[None]
    result = aligned_depth_to_source_ground_truth(prediction, affine, ground_truth)
    assert result["scale"] == 3.0
    assert result["valid_pixels"] == 4
    assert result["mean_abs_rel"] == 0.0
