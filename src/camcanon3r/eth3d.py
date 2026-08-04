"""Readers for the official ETH3D high-resolution multi-view format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import (
    aligned_depth_to_source_ground_truth,
    aligned_point_cloud_accuracy_completeness,
    focal_relative_error,
    pairwise_relative_pose_errors,
    principal_point_error,
)

_TARGET_POINT_CACHE: dict[tuple[object, ...], np.ndarray] = {}


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


def colmap_camera_matrix(camera: ColmapCamera) -> np.ndarray:
    """Return the pinhole component shared by supported COLMAP models."""

    if camera.model in {"PINHOLE", "OPENCV", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        if len(camera.parameters) < 4:
            raise ValueError(
                f"camera model {camera.model} requires fx, fy, cx, and cy"
            )
        fx, fy, cx, cy = camera.parameters[:4]
    elif camera.model in {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
    }:
        if len(camera.parameters) < 3:
            raise ValueError(f"camera model {camera.model} requires f, cx, and cy")
        focal, cx, cy = camera.parameters[:3]
        fx = fy = focal
    else:
        raise ValueError(f"unsupported COLMAP camera model: {camera.model}")
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _thin_prism_distort_and_jacobian(
    points: np.ndarray, parameters: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    if len(parameters) != 12:
        raise ValueError("THIN_PRISM_FISHEYE requires 12 parameters")
    k1, k2, p1, p2, k3, k4, sx1, sy1 = parameters[4:]
    x = points[:, 0]
    y = points[:, 1]
    x2 = x * x
    xy = x * y
    y2 = y * y
    r2 = x2 + y2
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
    distorted = np.column_stack(
        [
            x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2) + sx1 * r2,
            y * radial + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2) + sy1 * r2,
        ]
    )

    term1 = 2.0 * k1 + r2 * (4.0 * k2 + r2 * (6.0 * k3 + r2 * 8.0 * k4))
    term2 = radial
    term3 = xy * term1 + 2.0 * (p1 * x + p2 * y)
    jacobian = np.empty((len(points), 2, 2), dtype=np.float64)
    jacobian[:, 0, 0] = (
        x2 * term1 + term2 + 6.0 * p2 * x + 2.0 * p1 * y + 2.0 * sx1 * x
    )
    jacobian[:, 0, 1] = term3 + 2.0 * sx1 * y
    jacobian[:, 1, 0] = term3 + 2.0 * sy1 * x
    jacobian[:, 1, 1] = (
        y2 * term1 + term2 + 6.0 * p1 * y + 2.0 * p2 * x + 2.0 * sy1 * y
    )
    return distorted, jacobian


def colmap_unproject_pixels(
    camera: ColmapCamera, pixels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lift COLMAP pixels to normalized rays ``(x/z, y/z)``.

    The THIN_PRISM_FISHEYE path follows the ETH3D/COLMAP order: remove the
    polynomial/tangential/thin-prism deformation with Newton iterations, then
    invert the equidistant fisheye map. Failed or back-hemisphere lifts are
    marked invalid instead of silently approximated as pinhole rays.
    """

    image_points = np.asarray(pixels, dtype=np.float64)
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("pixels must have shape (N, 2)")
    if not np.isfinite(image_points).all():
        raise ValueError("pixels contain a non-finite value")
    intrinsics = colmap_camera_matrix(camera)
    distorted = np.column_stack(
        [
            (image_points[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0],
            (image_points[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1],
        ]
    )
    if camera.model in {"PINHOLE", "SIMPLE_PINHOLE"}:
        return distorted, np.ones(len(distorted), dtype=bool)
    if camera.model != "THIN_PRISM_FISHEYE":
        raise ValueError(
            f"point backprojection is unsupported for {camera.model}"
        )

    solution = distorted.copy()
    converged = np.zeros(len(solution), dtype=bool)
    valid = np.ones(len(solution), dtype=bool)
    for _ in range(100):
        active = valid & ~converged
        if not np.any(active):
            break
        distorted_candidate, jacobian = _thin_prism_distort_and_jacobian(
            solution[active], camera.parameters
        )
        residual = distorted_candidate - distorted[active]
        determinant = (
            jacobian[:, 0, 0] * jacobian[:, 1, 1]
            - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        )
        solvable = np.abs(determinant) > 1e-15
        active_indices = np.flatnonzero(active)
        valid[active_indices[~solvable]] = False
        if not np.any(solvable):
            continue
        selected_jacobian = jacobian[solvable]
        selected_residual = residual[solvable]
        selected_determinant = determinant[solvable]
        step = np.column_stack(
            [
                (
                    selected_jacobian[:, 1, 1] * selected_residual[:, 0]
                    - selected_jacobian[:, 0, 1] * selected_residual[:, 1]
                )
                / selected_determinant,
                (
                    -selected_jacobian[:, 1, 0] * selected_residual[:, 0]
                    + selected_jacobian[:, 0, 0] * selected_residual[:, 1]
                )
                / selected_determinant,
            ]
        )
        selected_indices = active_indices[solvable]
        radius_squared = np.maximum(
            np.sum(solution[selected_indices] ** 2, axis=1) * 0.01,
            0.01,
        )
        step_squared = np.sum(step**2, axis=1)
        too_large = step_squared > radius_squared
        step[too_large] *= np.sqrt(
            radius_squared[too_large] / step_squared[too_large]
        )[:, None]
        solution[selected_indices] -= step
        clipped_step_squared = np.sum(step**2, axis=1)
        converged[selected_indices[clipped_step_squared < 1e-10]] = True

    valid &= converged
    final_distorted, _ = _thin_prism_distort_and_jacobian(
        solution, camera.parameters
    )
    valid &= np.sum((final_distorted - distorted) ** 2, axis=1) < 1e-10
    theta = np.linalg.norm(solution, axis=1)
    valid &= theta < np.pi / 2.0
    scale = np.ones(len(solution), dtype=np.float64)
    nonzero = theta > np.finfo(np.float64).eps
    scale[nonzero] = np.tan(theta[nonzero]) / theta[nonzero]
    normalized = solution * scale[:, None]
    valid &= np.isfinite(normalized).all(axis=1)
    return normalized, valid


def _deterministic_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("maximum point count must be positive")
    if len(indices) <= maximum:
        return indices
    selected = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[selected]


def _depth_points_in_world(
    source_depth: np.ndarray,
    camera: ColmapCamera,
    extrinsic: np.ndarray,
    *,
    maximum_points: int,
) -> np.ndarray:
    valid_indices = np.flatnonzero(
        np.isfinite(source_depth.reshape(-1)) & (source_depth.reshape(-1) > 1e-8)
    )
    valid_indices = _deterministic_indices(valid_indices, maximum_points)
    if not len(valid_indices):
        return np.empty((0, 3), dtype=np.float64)
    pixels = np.column_stack(
        [valid_indices % camera.width, valid_indices // camera.width]
    )
    normalized, valid_lifts = colmap_unproject_pixels(camera, pixels)
    valid_indices = valid_indices[valid_lifts]
    normalized = normalized[valid_lifts]
    depth = np.asarray(source_depth.reshape(-1)[valid_indices], dtype=np.float64)
    camera_points = np.column_stack(
        [normalized[:, 0] * depth, normalized[:, 1] * depth, depth]
    )
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    return (camera_points - translation) @ rotation


def _target_point_cloud(
    source_depths: list[np.ndarray],
    camera: ColmapCamera,
    extrinsics: np.ndarray,
    *,
    maximum_points: int,
) -> np.ndarray:
    depth_paths = tuple(str(getattr(depth, "filename", "")) for depth in source_depths)
    key: tuple[object, ...] = (
        camera,
        extrinsics.tobytes(),
        depth_paths,
        maximum_points,
    )
    cached = _TARGET_POINT_CACHE.get(key)
    if cached is not None:
        return cached
    per_view_maximum = max(1, maximum_points)
    points = np.concatenate(
        [
            _depth_points_in_world(
                depth,
                camera,
                extrinsic,
                maximum_points=per_view_maximum,
            )
            for depth, extrinsic in zip(source_depths, extrinsics, strict=True)
        ]
    )
    if not len(points):
        raise ValueError("ETH3D depth maps contain no valid target points")
    _TARGET_POINT_CACHE[key] = points
    return points


def _world_point_stack(value: np.ndarray, view_count: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    while points.ndim > 4 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 4 or points.shape[0] != view_count or points.shape[-1] != 3:
        raise ValueError(
            "world points must reduce to shape "
            f"({view_count}, H, W, 3), got {points.shape}"
        )
    return points


def _predicted_gt_supported_points(
    world_points: np.ndarray,
    source_to_model: np.ndarray,
    source_depths: list[np.ndarray],
    *,
    maximum_points: int,
) -> np.ndarray:
    if len(source_depths) != len(world_points):
        raise ValueError("one source depth map is required for every point map")
    selected_points: list[np.ndarray] = []
    for view, point_map in enumerate(world_points):
        height, width = point_map.shape[:2]
        grid_y, grid_x = np.mgrid[:height, :width]
        pixels = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.ones(height * width)],
            axis=0,
        )
        source_pixels = np.linalg.solve(source_to_model[view], pixels)
        source_x = np.rint(source_pixels[0] / source_pixels[2]).astype(np.int64)
        source_y = np.rint(source_pixels[1] / source_pixels[2]).astype(np.int64)
        source_depth = source_depths[view]
        in_bounds = (
            (source_x >= 0)
            & (source_x < source_depth.shape[1])
            & (source_y >= 0)
            & (source_y < source_depth.shape[0])
        )
        sampled_depth = np.full(height * width, np.nan, dtype=np.float64)
        sampled_depth[in_bounds] = source_depth[
            source_y[in_bounds], source_x[in_bounds]
        ]
        flattened = point_map.reshape(-1, 3)
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 1e-8)
            & np.isfinite(flattened).all(axis=1)
        )
        indices = _deterministic_indices(np.flatnonzero(valid), maximum_points)
        selected_points.append(flattened[indices])
    points = np.concatenate(selected_points)
    if not len(points):
        raise ValueError("no GT-supported predicted world points remain")
    return points


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
        predicted_intrinsics = np.asarray(
            prediction["intrinsic"], dtype=np.float64
        )
        while predicted_intrinsics.ndim > 3 and predicted_intrinsics.shape[0] == 1:
            predicted_intrinsics = predicted_intrinsics[0]
        source_to_model = np.asarray(
            prediction["source_to_model_affine"], dtype=np.float64
        )
        while source_to_model.ndim > 3 and source_to_model.shape[0] == 1:
            source_to_model = source_to_model[0]
        expected_shape = (len(inputs), 3, 3)
        if predicted_intrinsics.shape != expected_shape:
            raise ValueError(
                "prediction intrinsics must have shape "
                f"{expected_shape}, got {predicted_intrinsics.shape}"
            )
        if source_to_model.shape != expected_shape:
            raise ValueError(
                "source-to-model affines must have shape "
                f"{expected_shape}, got {source_to_model.shape}"
            )
        source_intrinsics = np.linalg.solve(
            source_to_model, predicted_intrinsics
        )
        target_intrinsics = colmap_camera_matrix(camera)
        focal_errors = np.asarray(
            [
                focal_relative_error(intrinsics, target_intrinsics)
                for intrinsics in source_intrinsics
            ]
        )
        principal_point_errors = np.asarray(
            [
                principal_point_error(
                    intrinsics,
                    target_intrinsics,
                    (camera.width, camera.height),
                )
                for intrinsics in source_intrinsics
            ]
        )
        depth = None
        point_cloud = None
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
            predicted_points = _predicted_gt_supported_points(
                _world_point_stack(prediction["world_points"], len(inputs)),
                source_to_model,
                source_depths,
                maximum_points=100_000,
            )
            target_points = _target_point_cloud(
                source_depths,
                camera,
                ground_truth_extrinsics,
                maximum_points=100_000,
            )
            point_metadata = {
                "protocol": "raw_depth_backprojection",
                "per_view_pre_voxel_cap": 100_000,
                "predicted_gt_supported_points_before_voxel": len(
                    predicted_points
                ),
                "target_points_before_voxel": len(target_points),
            }
            try:
                point_cloud = aligned_point_cloud_accuracy_completeness(
                    predicted_points,
                    target_points,
                    prediction["extrinsic"],
                    ground_truth_extrinsics,
                    voxel_size=0.01,
                    maximum_points=100_000,
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
                    "accuracy_meters": _finite_summary(np.asarray([])),
                    "completeness_meters": _finite_summary(np.asarray([])),
                }
            else:
                point_cloud["status"] = "available"
            point_cloud.update(point_metadata)

    return {
        "prediction": str(prediction_path.resolve()),
        "variant": prediction_path.stem,
        "inputs": inputs,
        "camera_model": camera.model,
        "camera_size": [camera.width, camera.height],
        "intrinsics": {
            "focal_relative_error": _finite_summary(focal_errors),
            "principal_point_normalized_error": _finite_summary(
                principal_point_errors
            ),
            "per_view": [
                {
                    "input": name,
                    "focal_relative_error": float(focal_error),
                    "principal_point_normalized_error": float(
                        principal_point_error_value
                    ),
                }
                for name, focal_error, principal_point_error_value in zip(
                    inputs,
                    focal_errors,
                    principal_point_errors,
                    strict=True,
                )
            ],
        },
        "relative_rotation_degrees": _finite_summary(
            pose_errors["rotation_degrees"]
        ),
        "translation_direction_degrees": _finite_summary(
            pose_errors["translation_direction_degrees"]
        ),
        "depth": depth,
        "point_cloud": point_cloud,
    }
