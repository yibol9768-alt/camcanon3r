import json
from pathlib import Path

import numpy as np
import pytest

from camcanon3r.prediction import (
    save_npz_compressed_atomic,
    stack_equal_shapes,
    world_to_camera_from_camera_to_world,
    write_json_atomic,
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


def test_prediction_outputs_are_replaced_atomically(tmp_path: Path) -> None:
    archive = tmp_path / "scene" / "identity.npz"
    save_npz_compressed_atomic(archive, depth=np.asarray([1.0, 2.0]))
    with np.load(archive) as loaded:
        np.testing.assert_allclose(loaded["depth"], [1.0, 2.0])
    assert not archive.with_name(f".{archive.name}.tmp").exists()

    metadata = archive.with_suffix(".json")
    write_json_atomic(metadata, {"finite": 1.0})
    assert json.loads(metadata.read_text()) == {"finite": 1.0}
    assert not metadata.with_name(f".{metadata.name}.tmp").exists()


def test_atomic_json_rejects_nonfinite_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "prediction.json"
    with pytest.raises(ValueError, match="Out of range float"):
        write_json_atomic(metadata, {"invalid": float("nan")})
    assert not metadata.exists()
