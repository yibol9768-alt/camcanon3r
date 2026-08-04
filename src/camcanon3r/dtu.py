"""Readers and camera geometry for the official DTU MVS data set."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.linalg import rq
from scipy.spatial import cKDTree

from .metrics import (
    camera_pose_similarity,
    focal_relative_error,
    pairwise_relative_pose_errors,
    principal_point_error,
)

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
_CAMERA_ID = re.compile(r"^rect_(\d{3})_\d+_r5000$")


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


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": len(finite),
        "median": float(np.median(finite)) if len(finite) else None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
    }


def _stack(value: np.ndarray, shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    while array.ndim > len(shape_tail) + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(
            f"{label} must have shape (V, {shape_tail}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains a non-finite value")
    return array


def _world_points(value: np.ndarray, view_count: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    while points.ndim > 4 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 4 or points.shape[0] != view_count or points.shape[-1] != 3:
        raise ValueError(
            f"world_points must have shape ({view_count}, H, W, 3), got {points.shape}"
        )
    return points


def _deterministic_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("maximum point count must be positive")
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]


def _source_supported_world_points(
    world_points: np.ndarray,
    source_to_model: np.ndarray,
    source_sizes: list[tuple[int, int]],
    *,
    maximum_per_view: int,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for view, (point_map, affine, source_size) in enumerate(
        zip(world_points, source_to_model, source_sizes, strict=True)
    ):
        height, width = point_map.shape[:2]
        grid_y, grid_x = np.mgrid[:height, :width]
        pixels = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.ones(height * width)]
        )
        source_pixels = np.linalg.solve(affine, pixels)
        source_x = source_pixels[0] / source_pixels[2]
        source_y = source_pixels[1] / source_pixels[2]
        source_width, source_height = source_size
        flattened = point_map.reshape(-1, 3)
        valid = (
            (source_x >= 0.0)
            & (source_x <= source_width - 1)
            & (source_y >= 0.0)
            & (source_y <= source_height - 1)
            & np.isfinite(flattened).all(axis=1)
        )
        indices = _deterministic_indices(np.flatnonzero(valid), maximum_per_view)
        if len(indices):
            selected.append(flattened[indices])
        else:
            raise ValueError(f"DTU view {view} has no source-supported world points")
    return np.concatenate(selected)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def _load_observation_mask(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    payload = loadmat(path)
    mask = np.asarray(payload["ObsMask"], dtype=bool)
    bounding_box = np.asarray(payload["BB"], dtype=np.float64)
    resolution = float(np.asarray(payload["Res"]).squeeze())
    if mask.ndim != 3 or bounding_box.shape != (2, 3):
        raise ValueError(f"invalid DTU observation mask geometry: {path}")
    if (
        not np.isfinite(bounding_box).all()
        or not np.isfinite(resolution)
        or resolution <= 0
    ):
        raise ValueError(f"invalid DTU observation mask scale: {path}")
    return mask, bounding_box, resolution


def _observation_membership(
    points: np.ndarray,
    mask: np.ndarray,
    bounding_box: np.ndarray,
    resolution: float,
) -> np.ndarray:
    voxel = np.floor((points - bounding_box[0]) / resolution + 0.5).astype(np.int64)
    inside = np.all((voxel >= 0) & (voxel < np.asarray(mask.shape)), axis=1)
    selected = np.zeros(len(points), dtype=bool)
    valid = voxel[inside]
    selected[inside] = mask[valid[:, 0], valid[:, 1], valid[:, 2]]
    return selected


def _load_ground_plane(path: Path) -> np.ndarray:
    plane = np.asarray(loadmat(path)["P"], dtype=np.float64).reshape(-1)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError(f"invalid DTU ground plane: {path}")
    return plane


@lru_cache(maxsize=2)
def _dtu_target_resources(
    point_path: Path, mask_path: Path, plane_path: Path
) -> tuple[np.ndarray, cKDTree, np.ndarray, np.ndarray, float, np.ndarray]:
    target = read_ply_vertices(point_path)
    mask, bounding_box, resolution = _load_observation_mask(mask_path)
    return (
        target,
        cKDTree(target),
        mask,
        bounding_box,
        resolution,
        _load_ground_plane(plane_path),
    )


def _dtu_point_metrics(
    predicted_points: np.ndarray,
    predicted_extrinsics: np.ndarray,
    target_extrinsics: np.ndarray,
    *,
    point_path: Path,
    mask_path: Path,
    plane_path: Path,
    voxel_size_millimeters: float = 0.2,
    outlier_threshold_millimeters: float = 20.0,
    maximum_points: int = 100_000,
) -> dict[str, object]:
    similarity = camera_pose_similarity(predicted_extrinsics, target_extrinsics)
    aligned = float(similarity["scale"]) * (
        np.asarray(similarity["rotation"]) @ predicted_points.T
    ).T + np.asarray(similarity["translation"])
    aligned = _voxel_downsample(aligned, voxel_size_millimeters)
    (
        full_target,
        full_target_tree,
        mask,
        bounding_box,
        resolution,
        plane,
    ) = _dtu_target_resources(point_path, mask_path, plane_path)
    aligned = aligned[_observation_membership(aligned, mask, bounding_box, resolution)]
    if not len(aligned):
        raise ValueError("DTU observation mask removed every predicted point")
    aligned = aligned[_deterministic_indices(np.arange(len(aligned)), maximum_points)]

    target_above_plane = (
        np.column_stack([full_target, np.ones(len(full_target))]) @ plane > 0.0
    )
    target = full_target[target_above_plane]
    target = target[_deterministic_indices(np.arange(len(target)), maximum_points)]
    if not len(target):
        raise ValueError("DTU ground plane removed every target point")

    predicted_tree = cKDTree(aligned)
    accuracy = np.asarray(full_target_tree.query(aligned, workers=1)[0])
    completeness = np.asarray(predicted_tree.query(target, workers=1)[0])
    accuracy = accuracy[accuracy < outlier_threshold_millimeters]
    completeness = completeness[completeness < outlier_threshold_millimeters]
    if not len(accuracy) or not len(completeness):
        raise ValueError("DTU 20 mm outlier filter removed every distance")
    return {
        "status": "available",
        "protocol": "official_mask_plane_threshold_deterministic_cap",
        "alignment": {
            "source": "camera_pose_rotation_then_center_scale_translation",
            "scale": float(similarity["scale"]),
            "rotation": np.asarray(similarity["rotation"]).tolist(),
            "translation": np.asarray(similarity["translation"]).tolist(),
        },
        "voxel_size_millimeters": voxel_size_millimeters,
        "outlier_threshold_millimeters": outlier_threshold_millimeters,
        "maximum_points": maximum_points,
        "predicted_points_evaluated": len(aligned),
        "target_points_evaluated": len(target),
        "accuracy_millimeters": _finite_summary(accuracy),
        "completeness_millimeters": _finite_summary(completeness),
    }


def evaluate_dtu_prediction(
    prediction_path: Path,
    calibration_dir: Path,
    *,
    scan: int,
    gt_root: Path | None = None,
) -> dict[str, object]:
    """Evaluate DTU camera geometry and optional official-style point distances."""

    metadata = json.loads(prediction_path.with_suffix(".json").read_text())
    inputs = [str(value) for value in metadata["inputs"]]
    camera_ids: list[int] = []
    for name in inputs:
        match = _CAMERA_ID.fullmatch(Path(name).stem)
        if match is None:
            raise ValueError(
                f"cannot resolve DTU camera ID from prediction input: {name}"
            )
        camera_ids.append(int(match.group(1)))
    if len(camera_ids) < 2 or len(set(camera_ids)) != len(camera_ids):
        raise ValueError(
            "DTU prediction requires unique inputs from at least two cameras"
        )
    cameras = [
        read_dtu_projection(calibration_dir / f"pos_{camera_id:03d}.txt")
        for camera_id in camera_ids
    ]
    target_intrinsics = np.stack([camera[0] for camera in cameras])
    target_extrinsics = np.stack([camera[1] for camera in cameras])
    transforms = metadata.get("spatial_transforms")
    if (
        not isinstance(transforms, list)
        or [str(record["input"]) for record in transforms] != inputs
    ):
        raise ValueError("DTU prediction metadata has inconsistent spatial transforms")
    source_sizes = [
        tuple(int(value) for value in record["input_size"]) for record in transforms
    ]

    with np.load(prediction_path) as prediction:
        predicted_extrinsics = _stack(prediction["extrinsic"], (3, 4), "extrinsic")
        predicted_intrinsics = _stack(prediction["intrinsic"], (3, 3), "intrinsic")
        source_to_model = _stack(
            prediction["source_to_model_affine"],
            (3, 3),
            "source_to_model_affine",
        )
        if not (
            len(predicted_extrinsics)
            == len(predicted_intrinsics)
            == len(source_to_model)
            == len(inputs)
        ):
            raise ValueError("DTU prediction view counts do not match metadata")
        pose_errors = pairwise_relative_pose_errors(
            target_extrinsics, predicted_extrinsics
        )
        source_intrinsics = np.linalg.solve(source_to_model, predicted_intrinsics)
        focal_errors = np.asarray(
            [
                focal_relative_error(predicted, target)
                for predicted, target in zip(
                    source_intrinsics, target_intrinsics, strict=True
                )
            ]
        )
        principal_errors = np.asarray(
            [
                principal_point_error(predicted, target, size)
                for predicted, target, size in zip(
                    source_intrinsics,
                    target_intrinsics,
                    source_sizes,
                    strict=True,
                )
            ]
        )
        point_cloud = None
        if gt_root is not None:
            predicted_points = _source_supported_world_points(
                _world_points(prediction["world_points"], len(inputs)),
                source_to_model,
                source_sizes,
                maximum_per_view=100_000,
            )
            point_metadata = {
                "predicted_source_supported_points_before_alignment": len(
                    predicted_points
                )
            }
            try:
                point_cloud = _dtu_point_metrics(
                    predicted_points,
                    predicted_extrinsics,
                    target_extrinsics,
                    point_path=gt_root / f"Points/stl/stl{scan:03d}_total.ply",
                    mask_path=gt_root / f"ObsMask/ObsMask{scan}_10.mat",
                    plane_path=gt_root / f"ObsMask/Plane{scan}.mat",
                )
            except ValueError as error:
                reason = str(error)
                alignment_failure = (
                    "degenerate for Sim(3)" in reason
                    or "camera-pose Sim(3) has a non-positive scale" in reason
                )
                if not alignment_failure:
                    raise
                status = (
                    "undefined_degenerate_camera_center_alignment"
                    if "degenerate for Sim(3)" in reason
                    else "undefined_nonpositive_camera_pose_scale"
                )
                point_cloud = {
                    "status": status,
                    "reason": reason,
                    "protocol": ("official_mask_plane_threshold_deterministic_cap"),
                    "accuracy_millimeters": _finite_summary(np.asarray([])),
                    "completeness_millimeters": _finite_summary(np.asarray([])),
                }
            point_cloud.update(point_metadata)
    return {
        "prediction": str(prediction_path.resolve()),
        "variant": prediction_path.stem,
        "scan": scan,
        "inputs": inputs,
        "camera_model": "PINHOLE",
        "camera_size": list(source_sizes[0]) if len(set(source_sizes)) == 1 else None,
        "camera_ids": camera_ids,
        "camera_sizes": [list(size) for size in source_sizes],
        "intrinsics": {
            "focal_relative_error": _finite_summary(focal_errors),
            "principal_point_normalized_error": _finite_summary(principal_errors),
            "per_view": [
                {
                    "input": name,
                    "camera_id": camera_id,
                    "camera_size": list(size),
                    "focal_relative_error": float(focal_error),
                    "principal_point_normalized_error": float(principal_error_value),
                }
                for name, camera_id, size, focal_error, principal_error_value in zip(
                    inputs,
                    camera_ids,
                    source_sizes,
                    focal_errors,
                    principal_errors,
                    strict=True,
                )
            ],
        },
        "relative_rotation_degrees": _finite_summary(pose_errors["rotation_degrees"]),
        "translation_direction_degrees": _finite_summary(
            pose_errors["translation_direction_degrees"]
        ),
        "depth": None,
        "point_cloud": point_cloud,
    }
