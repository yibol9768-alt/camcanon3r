"""Readers and camera geometry for the official DTU MVS data set."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import rq

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def decompose_projection_matrix(
    projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a projective camera into positive-focal K and world-to-camera E."""

    matrix = np.asarray(projection, dtype=np.float64)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        raise ValueError("projection matrix must be finite with shape (3, 4)")
    left = matrix[:, :3]
    if np.linalg.matrix_rank(left) != 3:
        raise ValueError("projection matrix has a singular camera block")
    raw_intrinsic, raw_rotation = rq(left)
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for signs in product((-1.0, 1.0), repeat=3):
        diagonal = np.diag(signs)
        intrinsic = raw_intrinsic @ diagonal
        rotation = diagonal @ raw_rotation
        if np.linalg.det(rotation) <= 0.0 or abs(intrinsic[2, 2]) < 1e-12:
            continue
        intrinsic = intrinsic / intrinsic[2, 2]
        if intrinsic[0, 0] > 0.0 and intrinsic[1, 1] > 0.0:
            candidates.append((intrinsic, rotation))
    if len(candidates) != 1:
        raise ValueError(
            "projection matrix has no unique positive-focal proper-rotation decomposition"
        )
    intrinsic, rotation = candidates[0]
    camera_center = -np.linalg.solve(left, matrix[:, 3])
    translation = -(rotation @ camera_center)
    extrinsic = np.column_stack([rotation, translation])
    reconstructed = intrinsic @ extrinsic
    scale = float(np.sum(matrix * reconstructed) / np.sum(reconstructed**2))
    residual = np.linalg.norm(matrix - scale * reconstructed) / np.linalg.norm(matrix)
    if residual > 1e-8:
        raise ValueError(f"projection decomposition residual is too large: {residual}")
    return intrinsic, extrinsic


def read_dtu_projection(path: Path) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(f"DTU projection matrix must contain 3x4 values: {path}")
    return decompose_projection_matrix(matrix)


def read_ply_vertices(path: Path) -> np.ndarray:
    """Read finite x/y/z vertices from an ASCII or scalar binary PLY file."""

    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"file is not PLY: {path}")
        file_format: str | None = None
        vertex_count: int | None = None
        current_element: str | None = None
        preceding_element_count = 0
        properties: list[tuple[str, str]] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            try:
                fields = line.decode("ascii").strip().split()
            except UnicodeDecodeError as error:
                raise ValueError(f"PLY header is not ASCII: {path}") from error
            if not fields or fields[0] in {"comment", "obj_info"}:
                continue
            if fields[0] == "format":
                if len(fields) != 3 or fields[2] != "1.0":
                    raise ValueError(f"unsupported PLY format declaration: {fields}")
                file_format = fields[1]
            elif fields[0] == "element":
                if len(fields) != 3:
                    raise ValueError(f"invalid PLY element declaration: {fields}")
                current_element = fields[1]
                count = int(fields[2])
                if vertex_count is None and current_element != "vertex":
                    preceding_element_count += count
                if current_element == "vertex":
                    vertex_count = count
            elif fields[0] == "property" and current_element == "vertex":
                if len(fields) != 3 or fields[1] == "list":
                    raise ValueError("DTU PLY vertex list properties are unsupported")
                if fields[1] not in _PLY_TYPES:
                    raise ValueError(f"unsupported PLY vertex type: {fields[1]}")
                properties.append((fields[2], fields[1]))
            elif fields[0] == "end_header":
                break
        if file_format not in {
            "ascii",
            "binary_little_endian",
            "binary_big_endian",
        }:
            raise ValueError(f"unsupported PLY format: {file_format}")
        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"PLY contains no vertices: {path}")
        if preceding_element_count:
            raise ValueError("PLY elements before vertices are unsupported")
        names = [name for name, _ in properties]
        if not {"x", "y", "z"}.issubset(names):
            raise ValueError("PLY vertices must contain x, y, and z properties")

        if file_format == "ascii":
            coordinates = np.loadtxt(
                handle,
                dtype=np.float64,
                max_rows=vertex_count,
                usecols=tuple(names.index(axis) for axis in ("x", "y", "z")),
                ndmin=2,
            )
            if len(coordinates) != vertex_count:
                raise ValueError(
                    f"PLY vertex count mismatch: expected={vertex_count}, "
                    f"actual={len(coordinates)}"
                )
        else:
            byte_order = "<" if file_format == "binary_little_endian" else ">"
            dtype = np.dtype(
                [
                    (name, np.dtype(byte_order + _PLY_TYPES[data_type]))
                    for name, data_type in properties
                ]
            )
            records = np.fromfile(handle, dtype=dtype, count=vertex_count)
            if len(records) != vertex_count:
                raise ValueError(
                    f"PLY vertex count mismatch: expected={vertex_count}, "
                    f"actual={len(records)}"
                )
            coordinates = np.column_stack(
                [records[axis].astype(np.float64) for axis in ("x", "y", "z")]
            )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"PLY contains a non-finite vertex: {path}")
    return coordinates
