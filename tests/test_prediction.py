import numpy as np
import pytest

from camcanon3r.prediction import (
    stack_equal_shapes,
    world_to_camera_from_camera_to_world,
)


def test_camera_to_world_conversion_recovers_world_to_camera() -> None:
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, 3] = [1.0, 2.0, 3.0]
    extrinsics = world_to_camera_from_camera_to_world(poses)
    np.testing.assert_allclose(extrinsics[0], np.eye(4)[:3])
    np.testing.assert_allclose(extrinsics[1, :, 3], [-1.0, -2.0, -3.0])


def test_camera_to_world_conversion_rejects_invalid_homogeneous_pose() -> None:
    malformed = np.eye(4)[None]
    malformed[0, 3, 3] = 2.0
    with pytest.raises(ValueError, match="homogeneous"):
        world_to_camera_from_camera_to_world(malformed)


def test_stack_equal_shapes_accepts_dense_and_rejects_ragged() -> None:
    values = [np.ones((2, 3)), np.zeros((2, 3))]
    stacked = stack_equal_shapes(values, label="depth")
    assert stacked.shape == (2, 2, 3)
    with pytest.raises(ValueError, match="unequal shapes"):
        stack_equal_shapes(
            [np.ones((2, 3)), np.ones((3, 2))], label="depth"
        )
