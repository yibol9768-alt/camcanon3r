"""Readers for the official ETH3D high-resolution multi-view format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import (
    aligned_depth_to_source_ground_truth,
    pairwise_relative_pose_errors,
)


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    parameters: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    extrinsic: np.ndarray


def quaternion_to_rotation(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    norm = np.linalg.norm(quaternion)
    if norm <= 0.0:
        raise ValueError("camera quaternion has zero norm")
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float64) / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _data_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def read_colmap_cameras(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in _data_lines(path):
        fields = line.split()
        camera_id = int(fields[0])
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            parameters=tuple(float(value) for value in fields[4:]),
        )
    if not cameras:
        raise ValueError(f"no cameras found in {path}")
    return cameras


def read_colmap_images(path: Path) -> dict[str, ColmapImage]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    while lines and not lines[0]:
        lines.pop(0)
    if len(lines) % 2:
        raise ValueError("COLMAP images file must contain pose/observation line pairs")
    images: dict[str, ColmapImage] = {}
    for line in lines[::2]:
        if not line:
            raise ValueError("COLMAP pose line is unexpectedly empty")
        fields = line.split()
        image_id = int(fields[0])
        rotation = quaternion_to_rotation(tuple(float(value) for value in fields[1:5]))
        translation = np.asarray([float(value) for value in fields[5:8]])
        camera_id = int(fields[8])
        name = fields[9]
        images[Path(name).stem] = ColmapImage(
            image_id=image_id,
            camera_id=camera_id,
            name=name,
            extrinsic=np.concatenate([rotation, translation[:, None]], axis=1),
        )
    if not images:
        raise ValueError(f"no images found in {path}")
    return images


def open_eth3d_depth(path: Path, *, width: int, height: int) -> np.memmap:
    expected_bytes = width * height * np.dtype("<f4").itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"ETH3D depth size mismatch for {path}: expected {expected_bytes}, "
            f"got {path.stat().st_size}"
        )
    return np.memmap(path, mode="r", dtype="<f4", shape=(height, width))


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    return {
        "count": len(finite),
        "median": float(np.median(finite)) if len(finite) else None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
    }


def evaluate_eth3d_prediction(
    prediction_path: Path,
    calibration_dir: Path,
    *,
    depth_dir: Path | None,
) -> dict[str, object]:
    """Evaluate one prediction against ETH3D pose and optional raw-depth GT.

    ``depth_dir=None`` is the supported pose-only path for pre-undistorted
    images. Raw depth must only be paired with the original DSLR calibration.
    """

    metadata = json.loads(prediction_path.with_suffix(".json").read_text())
    cameras = read_colmap_cameras(calibration_dir / "cameras.txt")
    images = read_colmap_images(calibration_dir / "images.txt")
    inputs = metadata["inputs"]
    matched = [images[Path(name).stem] for name in inputs]
    ground_truth_extrinsics = np.stack([item.extrinsic for item in matched])
    camera = cameras[matched[0].camera_id]
    if any(item.camera_id != camera.camera_id for item in matched):
        raise RuntimeError("the selected ETH3D views do not share one camera model")

    with np.load(prediction_path) as prediction:
        pose_errors = pairwise_relative_pose_errors(
            ground_truth_extrinsics, prediction["extrinsic"]
        )
        depth = None
        if depth_dir is not None:
            source_depths = [
                open_eth3d_depth(
                    depth_dir / f"{Path(name).stem}.JPG",
                    width=camera.width,
                    height=camera.height,
                )
                for name in inputs
            ]
            depth = aligned_depth_to_source_ground_truth(
                prediction["depth"],
                prediction["source_to_model_affine"],
                source_depths,
            )

    return {
        "prediction": str(prediction_path.resolve()),
        "inputs": inputs,
        "camera_model": camera.model,
        "camera_size": [camera.width, camera.height],
        "relative_rotation_degrees": _finite_summary(
            pose_errors["rotation_degrees"]
        ),
        "translation_direction_degrees": _finite_summary(
            pose_errors["translation_direction_degrees"]
        ),
        "depth": depth,
    }
