"""Camera-constrained fusion of support-preserving reconstruction orbits.

Every orbit member predicts geometry in an arbitrary world Sim(3).  This
module estimates that nuisance gauge from the member and projected cameras,
warps point maps onto one registered model grid through the logged pixel
affines, and robustly fuses only source-supported correspondences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import map_coordinates

from .metrics import camera_centers_from_extrinsics, rotation_geodesic_degrees
from .orbit_projection import chordal_rotation_mean


def _view_stack(value: ArrayLike, tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    while array.ndim > len(tail) + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != len(tail) + 1 or array.shape[1:] != tail:
        raise ValueError(f"{label} must have shape (V, {tail}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains a non-finite value")
    return array


def _point_stack(value: ArrayLike, view_count: int, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    while points.ndim > 4 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 4 or points.shape[0] != view_count or points.shape[-1] != 3:
        raise ValueError(
            f"{label} must have shape ({view_count}, H, W, 3), got {points.shape}"
        )
    return points


def _confidence_stack(value: ArrayLike, expected: tuple[int, int, int]) -> np.ndarray:
    confidence = np.asarray(value, dtype=np.float64)
    while confidence.ndim > 3 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.shape != expected:
        raise ValueError(
            f"point confidence must have shape {expected}, got {confidence.shape}"
        )
    return confidence


def camera_world_similarity(
    source_extrinsics: ArrayLike,
    target_extrinsics: ArrayLike,
) -> dict[str, Any]:
    """Fit ``X_target = scale * rotation @ X_source + translation``.

    The rotation is estimated from camera orientations only.  Positive scale
    and translation then align camera centers in least squares.  Ground truth
    is neither accepted nor required.
    """

    source = _view_stack(source_extrinsics, (3, 4), "source extrinsics")
    target = _view_stack(target_extrinsics, (3, 4), "target extrinsics")
    if len(source) != len(target) or len(source) < 2:
        raise ValueError("camera similarity requires matching multi-view poses")
    source_rotations = source[:, :, :3]
    target_rotations = target[:, :, :3]
    candidates = np.einsum(
        "vji,vjk->vik",
        target_rotations,
        source_rotations,
    )
    rotation = chordal_rotation_mean(candidates)
    source_centers = camera_centers_from_extrinsics(source)
    target_centers = camera_centers_from_extrinsics(target)
    source_rotated = (rotation @ source_centers.T).T
    source_centered = source_rotated - np.mean(source_rotated, axis=0)
    target_centered = target_centers - np.mean(target_centers, axis=0)
    denominator = float(np.sum(source_centered**2))
    if denominator <= 1e-12:
        raise ValueError("source camera centers are degenerate for Sim(3) alignment")
    scale = float(np.sum(source_centered * target_centered) / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("camera-constrained Sim(3) has a non-positive scale")
    translation = np.mean(target_centers, axis=0) - scale * np.mean(
        source_rotated, axis=0
    )
    aligned_centers = scale * source_rotated + translation
    center_errors = np.linalg.norm(aligned_centers - target_centers, axis=1)
    rotation_errors = np.asarray(
        [
            rotation_geodesic_degrees(
                target_rotation @ rotation,
                source_rotation,
            )
            for source_rotation, target_rotation in zip(
                source_rotations, target_rotations, strict=True
            )
        ]
    )
    target_baselines = np.linalg.norm(
        target_centers[:, None, :] - target_centers[None, :, :], axis=-1
    )
    nonzero = target_baselines[np.triu_indices(len(target_centers), k=1)]
    nonzero = nonzero[nonzero > 1e-12]
    normalizer = float(np.median(nonzero)) if len(nonzero) else 1.0
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "median_rotation_residual_degrees": float(np.median(rotation_errors)),
        "maximum_rotation_residual_degrees": float(np.max(rotation_errors)),
        "median_center_residual_normalized": float(
            np.median(center_errors) / normalizer
        ),
        "maximum_center_residual_normalized": float(np.max(center_errors) / normalizer),
        "ground_truth_used": False,
    }


def apply_world_similarity(
    points: ArrayLike, similarity: Mapping[str, Any]
) -> np.ndarray:
    """Map a point array through a fitted camera-world similarity."""

    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("world points must end in XYZ")
    rotation = np.asarray(similarity["rotation"], dtype=np.float64)
    translation = np.asarray(similarity["translation"], dtype=np.float64)
    scale = float(similarity["scale"])
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("world similarity has invalid rotation or translation")
    return scale * np.einsum("ij,...j->...i", rotation, values) + translation


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        raise ValueError("weighted median has no finite positive-weight values")
    selected_values = values[finite]
    selected_weights = weights[finite]
    order = np.argsort(selected_values, kind="stable")
    cumulative = np.cumsum(selected_weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(selected_values[order[index]])


def fuse_source_intrinsics(
    intrinsics: ArrayLike,
    source_to_model_affines: ArrayLike,
    *,
    member_weights: ArrayLike,
    reference_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse intrinsics in source coordinates and return them on a reference grid."""

    matrices = np.asarray(intrinsics, dtype=np.float64)
    affines = np.asarray(source_to_model_affines, dtype=np.float64)
    weights = np.asarray(member_weights, dtype=np.float64)
    if matrices.ndim != 4 or matrices.shape[2:] != (3, 3):
        raise ValueError("intrinsics must have shape (members, views, 3, 3)")
    if affines.shape != matrices.shape:
        raise ValueError("intrinsic and affine stacks must have matching shapes")
    if weights.shape != (len(matrices),) or np.any(weights < 0.0):
        raise ValueError("intrinsic member weights are invalid")
    if not 0 <= reference_index < len(matrices):
        raise ValueError("intrinsic reference index is out of range")
    source = np.linalg.solve(affines, matrices)
    fused_source = np.empty(source.shape[1:], dtype=np.float64)
    for view in range(source.shape[1]):
        for row in range(3):
            for column in range(3):
                fused_source[view, row, column] = _weighted_median(
                    source[:, view, row, column], weights
                )
        if fused_source[view, 2, 2] == 0.0:
            raise ValueError("fused source intrinsic has zero homogeneous scale")
        fused_source[view] /= fused_source[view, 2, 2]
    fused_model = np.einsum("vij,vjk->vik", affines[reference_index], fused_source)
    return fused_source, fused_model


def _sample_map(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return map_coordinates(
            array,
            [y, x],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
    return np.stack(
        [
            map_coordinates(
                array[..., channel],
                [y, x],
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            for channel in range(array.shape[-1])
        ],
        axis=-1,
    )


def _source_support(
    mask: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
) -> np.ndarray:
    values = map_coordinates(
        np.asarray(mask),
        [source_y, source_x],
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return values > 0.5


def _geometric_median_field(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values).all(axis=-1) & np.isfinite(weights) & (weights > 0.0)
    base = np.where(valid, weights, 0.0)
    safe_values = np.where(valid[..., None], values, 0.0)
    totals = np.sum(base, axis=0)
    current = np.divide(
        np.sum(base[..., None] * safe_values, axis=0),
        totals[..., None],
        out=np.full(values.shape[1:], np.nan, dtype=np.float64),
        where=totals[..., None] > 0.0,
    )
    for _ in range(iterations):
        distances = np.linalg.norm(values - current[None, ...], axis=-1)
        inverse = np.where(valid, base / np.maximum(distances, 1e-8), 0.0)
        inverse_totals = np.sum(inverse, axis=0)
        updated = np.divide(
            np.sum(inverse[..., None] * safe_values, axis=0),
            inverse_totals[..., None],
            out=current.copy(),
            where=inverse_totals[..., None] > 0.0,
        )
        if np.nanmax(np.linalg.norm(updated - current, axis=-1), initial=0.0) < 1e-8:
            current = updated
            break
        current = updated
    return current, np.count_nonzero(valid, axis=0)


def fuse_orbit_geometry(
    members: Mapping[str, Mapping[str, ArrayLike]],
    *,
    projected_extrinsics: ArrayLike,
    member_order: Sequence[str],
    member_weights: Mapping[str, float],
    source_support_masks: Sequence[ArrayLike],
    reference_label: str = "center",
    minimum_members: int = 3,
    tile_rows: int = 64,
    geometric_median_iterations: int = 20,
    maximum_confidence_ratio: float = 20.0,
) -> dict[str, Any]:
    """Fuse a reconstruction orbit into one camera-consistent prediction."""

    labels = [str(label) for label in member_order]
    if set(members) != set(labels) or set(member_weights) != set(labels):
        raise ValueError("geometry members and weights must match member order")
    if reference_label not in labels:
        raise ValueError("geometry reference label is not an orbit member")
    if minimum_members < 2 or minimum_members > len(labels):
        raise ValueError("minimum geometry member count is invalid")
    if (
        tile_rows <= 0
        or geometric_median_iterations <= 0
        or maximum_confidence_ratio <= 0.0
    ):
        raise ValueError("geometry fusion iteration and tile counts must be positive")
    reference_index = labels.index(reference_label)
    target_extrinsics = _view_stack(
        projected_extrinsics, (3, 4), "projected extrinsics"
    )
    view_count = len(target_extrinsics)
    weights = np.asarray([float(member_weights[label]) for label in labels])
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or np.sum(weights) <= 0:
        raise ValueError("geometry member weights must be finite and non-negative")

    points: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    affines: list[np.ndarray] = []
    model_affines: list[np.ndarray] = []
    protocol_affines: list[np.ndarray] = []
    similarities: list[dict[str, Any]] = []
    for label in labels:
        member = members[label]
        extrinsic = _view_stack(member["extrinsic"], (3, 4), f"{label} extrinsic")
        if len(extrinsic) != view_count:
            raise ValueError("geometry member view counts differ")
        point_map = _point_stack(member["world_points"], view_count, label)
        confidence_field = (
            "world_points_conf" if "world_points_conf" in member else "depth_conf"
        )
        confidence = _confidence_stack(member[confidence_field], point_map.shape[:3])
        similarity = camera_world_similarity(extrinsic, target_extrinsics)
        points.append(apply_world_similarity(point_map, similarity))
        confidences.append(confidence)
        intrinsics.append(
            _view_stack(member["intrinsic"], (3, 3), f"{label} intrinsic")
        )
        affines.append(
            _view_stack(
                member["source_to_model_affine"],
                (3, 3),
                f"{label} source-to-model affine",
            )
        )
        model_affines.append(
            _view_stack(
                member["model_preprocess_affine"],
                (3, 3),
                f"{label} model affine",
            )
        )
        protocol_affines.append(
            _view_stack(
                member["protocol_affine"],
                (3, 3),
                f"{label} protocol affine",
            )
        )
        similarities.append(similarity)
    if len(source_support_masks) != view_count:
        raise ValueError("one source support mask is required per view")
    confidence_normalizers = np.empty((len(labels), view_count), dtype=np.float64)
    for member_index, confidence in enumerate(confidences):
        for view in range(view_count):
            finite = confidence[view][
                np.isfinite(confidence[view]) & (confidence[view] > 0.0)
            ]
            confidence_normalizers[member_index, view] = (
                float(np.median(finite)) if len(finite) else 1.0
            )

    affine_stack = np.stack(affines)
    intrinsic_source, intrinsic_model = fuse_source_intrinsics(
        np.stack(intrinsics),
        affine_stack,
        member_weights=weights,
        reference_index=reference_index,
    )
    reference_points = points[reference_index]
    output_points = np.full(reference_points.shape, np.nan, dtype=np.float64)
    output_confidence = np.zeros(reference_points.shape[:3], dtype=np.float64)
    effective_counts = np.zeros(reference_points.shape[:3], dtype=np.int16)
    dispersion_values: list[np.ndarray] = []
    for view in range(view_count):
        height, width = reference_points[view].shape[:2]
        reference_affine_inverse = np.linalg.inv(affine_stack[reference_index, view])
        for first_row in range(0, height, tile_rows):
            last_row = min(height, first_row + tile_rows)
            grid_y, grid_x = np.mgrid[first_row:last_row, :width]
            flat = np.stack(
                [grid_x.reshape(-1), grid_y.reshape(-1), np.ones(grid_x.size)]
            )
            source_pixels = reference_affine_inverse @ flat
            source_x = source_pixels[0] / source_pixels[2]
            source_y = source_pixels[1] / source_pixels[2]
            support = _source_support(
                np.asarray(source_support_masks[view]), source_x, source_y
            )
            sampled_points = []
            sampled_weights = []
            for member_index in range(len(labels)):
                member_pixels = affine_stack[member_index, view] @ source_pixels
                member_x = member_pixels[0] / member_pixels[2]
                member_y = member_pixels[1] / member_pixels[2]
                sampled = _sample_map(points[member_index][view], member_x, member_y)
                confidence = _sample_map(
                    confidences[member_index][view], member_x, member_y
                )
                normalized_confidence = np.where(
                    np.isfinite(confidence) & (confidence > 0.0),
                    np.clip(
                        confidence / confidence_normalizers[member_index, view],
                        0.0,
                        maximum_confidence_ratio,
                    ),
                    0.0,
                )
                normalized_confidence[~support] = 0.0
                sampled_points.append(sampled)
                sampled_weights.append(weights[member_index] * normalized_confidence)
            value_stack = np.stack(sampled_points)
            weight_stack = np.stack(sampled_weights)
            fused, counts = _geometric_median_field(
                value_stack,
                weight_stack,
                iterations=geometric_median_iterations,
            )
            valid = (
                support & (counts >= minimum_members) & np.isfinite(fused).all(axis=1)
            )
            fused[~valid] = np.nan
            tile_shape = (last_row - first_row, width)
            output_points[view, first_row:last_row] = fused.reshape(*tile_shape, 3)
            effective_counts[view, first_row:last_row] = counts.reshape(tile_shape)
            confidence_sum = np.sum(weight_stack, axis=0)
            confidence_sum[~valid] = 0.0
            output_confidence[view, first_row:last_row] = confidence_sum.reshape(
                tile_shape
            )
            distances = np.linalg.norm(value_stack - fused[None, ...], axis=-1)
            distances[~np.isfinite(distances)] = np.nan
            if np.any(valid):
                dispersion_values.append(np.nanmedian(distances[:, valid], axis=0))

    rotations = target_extrinsics[:, :, :3]
    translations = target_extrinsics[:, :, 3]
    camera_points = np.einsum("vij,vhwj->vhwi", rotations, output_points)
    camera_points += translations[:, None, None, :]
    depth = camera_points[..., 2]
    depth[~np.isfinite(output_points).all(axis=-1)] = np.nan
    finite_dispersion = (
        np.concatenate([value for value in dispersion_values if len(value)])
        if any(len(value) for value in dispersion_values)
        else np.asarray([])
    )
    return {
        "extrinsic": target_extrinsics,
        "intrinsic": intrinsic_model,
        "source_intrinsic": intrinsic_source,
        "depth": depth.astype(np.float32),
        "depth_conf": output_confidence.astype(np.float32),
        "world_points": output_points.astype(np.float32),
        "world_points_conf": output_confidence.astype(np.float32),
        "model_preprocess_affine": np.stack(model_affines)[reference_index],
        "protocol_affine": np.stack(protocol_affines)[reference_index],
        "source_to_model_affine": affine_stack[reference_index],
        "reference_label": reference_label,
        "minimum_members": minimum_members,
        "maximum_confidence_ratio": maximum_confidence_ratio,
        "effective_member_count": effective_counts,
        "valid_fused_pixels": int(np.count_nonzero(np.isfinite(depth))),
        "total_reference_pixels": int(depth.size),
        "valid_fused_fraction": float(
            np.count_nonzero(np.isfinite(depth)) / depth.size
        ),
        "median_point_dispersion": (
            float(np.median(finite_dispersion)) if len(finite_dispersion) else None
        ),
        "member_similarity": {
            label: {
                **{
                    key: value
                    for key, value in similarity.items()
                    if key not in {"rotation", "translation"}
                },
                "rotation": np.asarray(similarity["rotation"]).tolist(),
                "translation": np.asarray(similarity["translation"]).tolist(),
            }
            for label, similarity in zip(labels, similarities, strict=True)
        },
        "camera_constrained": True,
        "common_source_support_only": True,
        "ground_truth_used": False,
    }
