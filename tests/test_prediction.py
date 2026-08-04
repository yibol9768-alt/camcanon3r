import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from camcanon3r.prediction import (
    PREDICTION_SCHEMA_VERSION,
    input_sha256_records,
    save_npz_compressed_atomic,
    stack_equal_shapes,
    validate_prediction_pair,
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
        stack_equal_shapes([np.ones((2, 3)), np.ones((3, 2))], label="depth")


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


def _complete_prediction_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    prepared = tmp_path / "prepared" / "identity"
    prepared.mkdir(parents=True)
    image_paths = [prepared / "000.png", prepared / "001.png"]
    for index, path in enumerate(image_paths):
        Image.new("RGB", (8, 6), color=(index, 0, 0)).save(path)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    output = tmp_path / "predictions" / "identity.npz"
    affines = np.repeat(np.eye(3)[None], 2, axis=0)
    save_npz_compressed_atomic(
        output,
        extrinsic=np.repeat(np.eye(4)[None, :3], 2, axis=0),
        intrinsic=affines,
        depth=np.ones((2, 2, 2)),
        world_points=np.ones((2, 2, 2, 3)),
        model_preprocess_affine=affines,
        protocol_affine=affines,
        source_to_model_affine=affines,
    )
    write_json_atomic(
        output.with_suffix(".json"),
        {
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "scene_directory": str(prepared.resolve()),
            "inputs": [path.name for path in image_paths],
            "input_sha256": input_sha256_records(image_paths),
            "weights": str(weights.resolve()),
            "seed": 17,
            "spatial_transforms": [
                {
                    "input": path.name,
                    "input_size": [8, 6],
                    "model_preprocess_affine": np.eye(3).tolist(),
                    "protocol_affine": np.eye(3).tolist(),
                    "source_to_model_affine": np.eye(3).tolist(),
                }
                for path in image_paths
            ],
        },
    )
    return prepared, output, weights


def test_prediction_resume_validation_binds_inputs_and_affines(tmp_path: Path) -> None:
    prepared, output, weights = _complete_prediction_pair(tmp_path)
    metadata = validate_prediction_pair(
        output, prepared, max_views=2, seed=17, weights=weights
    )
    assert metadata["prediction_schema_version"] == PREDICTION_SCHEMA_VERSION

    Image.new("RGB", (8, 6), color=(99, 0, 0)).save(prepared / "000.png")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_prediction_pair(
            output, prepared, max_views=2, seed=17, weights=weights
        )


def test_prediction_resume_validation_rejects_corrupt_archive(tmp_path: Path) -> None:
    prepared, output, weights = _complete_prediction_pair(tmp_path)
    output.write_bytes(output.read_bytes()[:-32])
    with pytest.raises(ValueError, match="archive"):
        validate_prediction_pair(
            output, prepared, max_views=2, seed=17, weights=weights
        )
