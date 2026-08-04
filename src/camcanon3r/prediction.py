"""Model-neutral prediction archive helpers."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from .protocol import list_images

PREDICTION_SCHEMA_VERSION = "1.1"


def save_npz_compressed_atomic(path: Path, **arrays: ArrayLike) -> None:
    """Write a compressed prediction archive without exposing partial output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def input_sha256_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    """Record ordered input identities without embedding source images."""

    return [
        {"input": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]


def _view_stack(value: ArrayLike, tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > len(tail) + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != len(tail) + 1 or array.shape[1:] != tail:
        raise ValueError(f"{label} must have shape (V, {tail}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains a non-finite value")
    return array


def validate_prediction_pair(
    path: Path,
    prepared_dir: Path,
    *,
    max_views: int,
    seed: int,
    weights: Path,
) -> dict[str, object]:
    """Validate a completed prediction before a resumable sweep skips it.

    Legacy records remain resumable after structural checks. Records written by
    schema 1.1 additionally bind the exact input bytes, preventing a same-name
    prepared image replacement from silently reusing stale predictions.
    """

    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"prediction pair is incomplete: {path} and {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"prediction metadata is unreadable: {metadata_path}"
        ) from error
    if not isinstance(metadata, dict):
        raise TypeError(f"prediction metadata is not an object: {metadata_path}")

    image_paths = list_images(prepared_dir, max_views=max_views)
    inputs = [path.name for path in image_paths]
    if metadata.get("inputs") != inputs:
        raise ValueError(
            "prediction input order does not match the prepared scene: "
            f"expected={inputs}, actual={metadata.get('inputs')}"
        )
    if metadata.get("scene_directory") != str(prepared_dir.resolve()):
        raise ValueError("prediction scene directory does not match the prepared scene")
    if metadata.get("weights") != str(weights.resolve()):
        raise ValueError("prediction weights path does not match the requested weights")
    if metadata.get("seed") != seed:
        raise ValueError(
            f"prediction seed mismatch: expected={seed}, actual={metadata.get('seed')}"
        )

    transforms = metadata.get("spatial_transforms")
    if (
        not isinstance(transforms, list)
        or [
            record.get("input") if isinstance(record, dict) else None
            for record in transforms
        ]
        != inputs
    ):
        raise ValueError("prediction spatial transforms do not match the input order")
    for image_path, record in zip(image_paths, transforms, strict=True):
        with Image.open(image_path) as opened:
            actual_size = list(opened.size)
        if record.get("input_size") != actual_size:
            raise ValueError(
                f"prediction input size changed for {image_path.name}: "
                f"expected={actual_size}, actual={record.get('input_size')}"
            )

    schema = metadata.get("prediction_schema_version")
    if schema is not None and schema != PREDICTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported prediction schema version: {schema}")
    if schema == PREDICTION_SCHEMA_VERSION:
        expected_hashes = input_sha256_records(image_paths)
        if metadata.get("input_sha256") != expected_hashes:
            raise ValueError(
                "prediction input SHA-256 records do not match prepared inputs"
            )

    try:
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"prediction archive is unreadable: {path}") from error
    if corrupt_member is not None:
        raise ValueError(
            f"prediction archive has a corrupt member: {path}:{corrupt_member}"
        )

    required_arrays = {
        "extrinsic",
        "intrinsic",
        "depth",
        "world_points",
        "model_preprocess_affine",
        "protocol_affine",
        "source_to_model_affine",
    }
    try:
        with np.load(path, allow_pickle=False) as prediction:
            missing = required_arrays - set(prediction.files)
            if missing:
                raise ValueError(
                    f"prediction archive is missing arrays: {sorted(missing)}"
                )
            extrinsic = _view_stack(prediction["extrinsic"], (3, 4), "extrinsic")
            intrinsic = _view_stack(prediction["intrinsic"], (3, 3), "intrinsic")
            model_affine = _view_stack(
                prediction["model_preprocess_affine"], (3, 3), "model affine"
            )
            protocol_affine = _view_stack(
                prediction["protocol_affine"], (3, 3), "protocol affine"
            )
            combined_affine = _view_stack(
                prediction["source_to_model_affine"], (3, 3), "combined affine"
            )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "prediction archive"
        ):
            raise
        raise ValueError(f"prediction archive arrays are unreadable: {path}") from error
    view_count = len(inputs)
    if not all(
        len(array) == view_count
        for array in (
            extrinsic,
            intrinsic,
            model_affine,
            protocol_affine,
            combined_affine,
        )
    ):
        raise ValueError("prediction array view counts do not match metadata inputs")
    if not np.allclose(
        combined_affine,
        model_affine @ protocol_affine,
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ValueError("stored source-to-model affine is not the logged composition")
    for index, record in enumerate(transforms):
        for key, actual in (
            ("model_preprocess_affine", model_affine[index]),
            ("protocol_affine", protocol_affine[index]),
            ("source_to_model_affine", combined_affine[index]),
        ):
            try:
                logged = np.asarray(record[key], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"prediction transform {index} has no valid {key}"
                ) from error
            if logged.shape != (3, 3) or not np.allclose(
                logged, actual, rtol=1e-7, atol=1e-7
            ):
                raise ValueError(
                    f"prediction transform {index} disagrees with archive field {key}"
                )
    return metadata


def world_to_camera_from_camera_to_world(cam2world: ArrayLike) -> np.ndarray:
    """Convert a finite batch of 4x4 camera poses to 3x4 extrinsics."""

    poses = np.asarray(cam2world, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"camera-to-world poses must have shape (V, 4, 4), got {poses.shape}"
        )
    if not np.isfinite(poses).all():
        raise ValueError("camera-to-world poses contain a non-finite value")
    expected_last_row = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (len(poses), 4))
    if not np.allclose(poses[:, 3], expected_last_row, atol=1e-6):
        raise ValueError("camera-to-world poses are not homogeneous rigid transforms")
    return np.linalg.inv(poses)[:, :3, :4]


def stack_equal_shapes(values: Iterable[ArrayLike], *, label: str) -> np.ndarray:
    """Stack per-view arrays while rejecting ambiguous ragged archives."""

    arrays = [np.asarray(value) for value in values]
    if not arrays:
        raise ValueError(f"{label} must contain at least one view")
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"{label} views have unequal shapes: {sorted(shapes)}")
    result = np.stack(arrays)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result
