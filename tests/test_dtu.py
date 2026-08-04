from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from camcanon3r.dtu import (
    decompose_projection_matrix,
    read_dtu_projection,
    read_ply_vertices,
)


def test_decompose_projection_matrix_recovers_camera_up_to_scale(
    tmp_path: Path,
) -> None:
    intrinsic = np.array([[1200.0, 3.0, 800.0], [0.0, 1180.0, 600.0], [0.0, 0.0, 1.0]])
    rotation = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
    translation = np.array([20.0, -30.0, 900.0])
    projection = -2.5 * intrinsic @ np.column_stack([rotation, translation])

    actual_intrinsic, actual_extrinsic = decompose_projection_matrix(projection)
    assert actual_intrinsic == pytest.approx(intrinsic)
    assert actual_extrinsic[:, :3] == pytest.approx(rotation)
    assert actual_extrinsic[:, 3] == pytest.approx(translation)

    path = tmp_path / "pos_023.txt"
    np.savetxt(path, projection)
    file_intrinsic, file_extrinsic = read_dtu_projection(path)
    assert file_intrinsic == pytest.approx(intrinsic)
    assert file_extrinsic == pytest.approx(actual_extrinsic)


def test_read_ply_vertices_supports_ascii_and_binary(tmp_path: Path) -> None:
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ascii_path = tmp_path / "ascii.ply"
    ascii_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "end_header\n"
        "1 2 3 20\n"
        "4 5 6 30\n",
        encoding="ascii",
    )
    assert read_ply_vertices(ascii_path) == pytest.approx(expected)

    binary_path = tmp_path / "binary.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "end_header\n"
    ).encode("ascii")
    with binary_path.open("wb") as handle:
        handle.write(header)
        for point, red in zip(expected, (20, 30), strict=True):
            handle.write(struct.pack("<fffB", *point, red))
    assert read_ply_vertices(binary_path) == pytest.approx(expected)
