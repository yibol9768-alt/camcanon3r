"""Deterministic camera-aligned point-map rendering for qualitative evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _stack_world_points(value: np.ndarray, view_count: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    while points.ndim > 4 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 4 or points.shape[0] != view_count or points.shape[-1] != 3:
        raise ValueError(
            "world points must reduce to shape "
            f"({view_count}, H, W, 3), got {points.shape}"
        )
    return points


def _stack_affines(value: np.ndarray, view_count: int) -> np.ndarray:
    affines = np.asarray(value, dtype=np.float64)
    while affines.ndim > 3 and affines.shape[0] == 1:
        affines = affines[0]
    if affines.shape != (view_count, 3, 3) or not np.isfinite(affines).all():
        raise ValueError(
            f"source-to-model affines must have shape ({view_count}, 3, 3)"
        )
    return affines


def source_supported_prediction_points(
    world_points: np.ndarray,
    source_to_model: np.ndarray,
    source_sizes: Sequence[tuple[int, int]],
    *,
    maximum_per_view: int,
) -> np.ndarray:
    """Select finite point-map cells that map inside the source image canvas."""

    if maximum_per_view <= 0:
        raise ValueError("maximum points per view must be positive")
    sizes = [(int(width), int(height)) for width, height in source_sizes]
    points = _stack_world_points(world_points, len(sizes))
    affines = _stack_affines(source_to_model, len(sizes))
    selected: list[np.ndarray] = []
    for point_map, affine, (source_width, source_height) in zip(
        points, affines, sizes, strict=True
    ):
        if source_width <= 0 or source_height <= 0:
            raise ValueError("source image sizes must be positive")
        height, width = point_map.shape[:2]
        grid_y, grid_x = np.mgrid[:height, :width]
        pixels = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.ones(height * width)]
        )
        source_pixels = np.linalg.solve(affine, pixels)
        source_x = source_pixels[0] / source_pixels[2]
        source_y = source_pixels[1] / source_pixels[2]
        flattened = point_map.reshape(-1, 3)
        valid = (
            (source_x >= 0.0)
            & (source_x <= source_width - 1)
            & (source_y >= 0.0)
            & (source_y <= source_height - 1)
            & np.isfinite(flattened).all(axis=1)
        )
        indices = np.flatnonzero(valid)
        if len(indices) > maximum_per_view:
            indices = indices[
                np.linspace(0, len(indices) - 1, maximum_per_view, dtype=np.int64)
            ]
        if len(indices):
            selected.append(flattened[indices])
    if not selected:
        raise ValueError("prediction contains no finite source-supported points")
    return np.concatenate(selected)


def apply_camera_pose_alignment(
    points: np.ndarray, alignment: Mapping[str, object]
) -> np.ndarray:
    """Apply the evaluator's recorded orientation-preserving Sim(3)."""

    values = np.asarray(points, dtype=np.float64)
    scale = float(alignment["scale"])
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["translation"], dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(values).all()
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("qualitative camera-pose alignment is invalid")
    return scale * (rotation @ values.T).T + translation


def median_camera_baseline(extrinsics: np.ndarray) -> float:
    """Return a pose-only scale unit from the median pairwise camera baseline."""

    values = np.asarray(extrinsics, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 4) or len(values) < 2:
        raise ValueError("at least two 3x4 target extrinsics are required")
    if not np.isfinite(values).all():
        raise ValueError("target extrinsics contain a non-finite value")
    centers = np.stack(
        [
            -rotation.T @ translation
            for rotation, translation in zip(
                values[:, :, :3], values[:, :, 3], strict=True
            )
        ]
    )
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    upper = distances[np.triu_indices(len(centers), k=1)]
    positive = upper[upper > 1e-12]
    if not len(positive):
        raise ValueError("target camera centers have no positive baseline")
    return float(np.median(positive))


def rasterize_aligned_points(
    points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    source_size: tuple[int, int],
    *,
    output_size: tuple[int, int],
    baseline: float,
) -> np.ndarray:
    """Z-buffer aligned points into the first target-camera image plane.

    The returned finite pixels are log10 depth measured in target-camera
    baselines.  Projection axes and the depth normalization therefore do not
    depend on a model outcome or a per-scene point-cloud bounding box.
    """

    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(extrinsic, dtype=np.float64)
    source_width, source_height = (int(value) for value in source_size)
    output_width, output_height = (int(value) for value in output_size)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("aligned qualitative points must have shape (N, 3)")
    if matrix.shape != (3, 3) or pose.shape != (3, 4):
        raise ValueError("qualitative target camera matrices have invalid shapes")
    if (
        source_width <= 0
        or source_height <= 0
        or output_width <= 0
        or output_height <= 0
        or not np.isfinite(baseline)
        or baseline <= 0.0
    ):
        raise ValueError("qualitative raster dimensions and baseline must be positive")

    camera_points = (pose[:, :3] @ values.T).T + pose[:, 3]
    depth = camera_points[:, 2]
    in_front = np.isfinite(depth) & (depth > 1e-12)
    camera_points = camera_points[in_front]
    depth = depth[in_front]
    if not len(depth):
        raise ValueError("aligned points are all behind the target camera")
    homogeneous = (matrix @ camera_points.T).T
    x = homogeneous[:, 0] / homogeneous[:, 2]
    y = homogeneous[:, 1] / homogeneous[:, 2]
    x = np.floor(x * output_width / source_width).astype(np.int64)
    y = np.floor(y * output_height / source_height).astype(np.int64)
    inside = (x >= 0) & (x < output_width) & (y >= 0) & (y < output_height)
    x = x[inside]
    y = y[inside]
    depth = depth[inside]
    if not len(depth):
        raise ValueError("aligned points project outside the target image")
    flat = np.full(output_width * output_height, np.inf, dtype=np.float64)
    np.minimum.at(flat, y * output_width + x, depth)
    image = flat.reshape(output_height, output_width)
    valid = np.isfinite(image)
    image[valid] = np.log10(image[valid] / baseline)
    image[~valid] = np.nan
    return image
