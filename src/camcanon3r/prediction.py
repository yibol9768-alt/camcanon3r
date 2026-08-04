"""Model-neutral prediction archive helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


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


def world_to_camera_from_camera_to_world(cam2world: ArrayLike) -> np.ndarray:
    """Convert a finite batch of 4x4 camera poses to 3x4 extrinsics."""

    poses = np.asarray(cam2world, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"camera-to-world poses must have shape (V, 4, 4), got {poses.shape}")
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
