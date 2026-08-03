"""Readers for the official ETH3D high-resolution multi-view format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
