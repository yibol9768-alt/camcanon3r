"""Camera-space metrics used by the CamCanon3R protocol."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _rotation3(value: ArrayLike) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation matrix, got {rotation.shape}")
    if not np.isfinite(rotation).all():
        raise ValueError("rotation contains a non-finite value")
    return rotation


def _extrinsics34(value: ArrayLike) -> np.ndarray:
    extrinsics = np.asarray(value, dtype=np.float64)
    while extrinsics.ndim > 3 and extrinsics.shape[0] == 1:
        extrinsics = extrinsics[0]
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(
            "extrinsics must have shape (views, 3, 4) or (views, 4, 4), "
            f"got {extrinsics.shape}"
        )
    if not np.isfinite(extrinsics).all():
        raise ValueError("extrinsics contain a non-finite value")
    return extrinsics[:, :3, :4]


def focal_relative_error(predicted: ArrayLike, target: ArrayLike) -> float:
    predicted_k = np.asarray(predicted, dtype=np.float64)
    target_k = np.asarray(target, dtype=np.float64)
    ratios = np.abs(predicted_k[[0, 1], [0, 1]] - target_k[[0, 1], [0, 1]])
    ratios /= np.abs(target_k[[0, 1], [0, 1]])
    return float(np.mean(ratios))


def principal_point_error(
    predicted: ArrayLike, target: ArrayLike, image_size: tuple[int, int]
) -> float:
    predicted_k = np.asarray(predicted, dtype=np.float64)
    target_k = np.asarray(target, dtype=np.float64)
    diagonal = float(np.hypot(*image_size))
    return float(np.linalg.norm(predicted_k[:2, 2] - target_k[:2, 2]) / diagonal)


def rotation_geodesic_degrees(predicted: ArrayLike, target: ArrayLike) -> float:
    predicted_r = _rotation3(predicted)
    target_r = _rotation3(target)
    relative = predicted_r @ target_r.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def translation_direction_degrees(
    predicted: ArrayLike, target: ArrayLike, *, epsilon: float = 1e-12
) -> float:
    """Angle between two translation or baseline directions.

    Direction error is invariant to positive scale but not to a sign flip. A
    zero-length direction has no defined orientation and is reported as NaN.
    """

    predicted_t = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target_t = np.asarray(target, dtype=np.float64).reshape(-1)
    if predicted_t.shape != (3,) or target_t.shape != (3,):
        raise ValueError("translation directions must each contain three values")
    if not np.isfinite(predicted_t).all() or not np.isfinite(target_t).all():
        raise ValueError("translation direction contains a non-finite value")
    denominator = np.linalg.norm(predicted_t) * np.linalg.norm(target_t)
    if denominator <= epsilon:
        return float("nan")
    cosine = np.clip(float(predicted_t @ target_t) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pairwise_relative_pose_errors(
    reference: ArrayLike, candidate: ArrayLike
) -> dict[str, np.ndarray]:
    """Compare two world-to-camera pose sets without a global alignment.

    Relative rotations and camera-center baselines expressed in the first
    camera of each pair cancel the arbitrary world frame. Translation scale
    also cancels because only the baseline direction is compared.
    """

    reference_e = _extrinsics34(reference)
    candidate_e = _extrinsics34(candidate)
    if reference_e.shape != candidate_e.shape:
        raise ValueError(
            "reference and candidate extrinsics must have the same shape, "
            f"got {reference_e.shape} and {candidate_e.shape}"
        )
    if len(reference_e) < 2:
        raise ValueError("at least two camera views are required")

    reference_r = reference_e[:, :, :3]
    candidate_r = candidate_e[:, :, :3]
    reference_t = reference_e[:, :, 3]
    candidate_t = candidate_e[:, :, 3]
    reference_centers = -np.einsum("vji,vj->vi", reference_r, reference_t)
    candidate_centers = -np.einsum("vji,vj->vi", candidate_r, candidate_t)

    pairs: list[tuple[int, int]] = []
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for first in range(len(reference_e) - 1):
        for second in range(first + 1, len(reference_e)):
            pairs.append((first, second))
            reference_relative_r = reference_r[second] @ reference_r[first].T
            candidate_relative_r = candidate_r[second] @ candidate_r[first].T
            rotation_errors.append(
                rotation_geodesic_degrees(candidate_relative_r, reference_relative_r)
            )

            reference_baseline = reference_r[first] @ (
                reference_centers[second] - reference_centers[first]
            )
            candidate_baseline = candidate_r[first] @ (
                candidate_centers[second] - candidate_centers[first]
            )
            translation_errors.append(
                translation_direction_degrees(candidate_baseline, reference_baseline)
            )

    return {
        "pairs": np.asarray(pairs, dtype=np.int64),
        "rotation_degrees": np.asarray(rotation_errors, dtype=np.float64),
        "translation_direction_degrees": np.asarray(
            translation_errors, dtype=np.float64
        ),
    }


def _depth_stack(value: ArrayLike) -> np.ndarray:
    depths = np.asarray(value, dtype=np.float64)
    while depths.ndim > 3 and depths.shape[0] == 1:
        depths = depths[0]
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    if depths.ndim != 3:
        raise ValueError(f"depth must reduce to shape (views, H, W), got {depths.shape}")
    return depths


def _affine_stack(value: ArrayLike, view_count: int) -> np.ndarray:
    affines = np.asarray(value, dtype=np.float64)
    while affines.ndim > 3 and affines.shape[0] == 1:
        affines = affines[0]
    if affines.shape != (view_count, 3, 3):
        raise ValueError(
            f"expected {view_count} affine matrices with shape 3x3, got {affines.shape}"
        )
    if not np.isfinite(affines).all():
        raise ValueError("affines contain a non-finite value")
    return affines


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    valid = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    clipped_x = np.clip(x, 0.0, width - 1)
    clipped_y = np.clip(y, 0.0, height - 1)
    x0 = np.floor(clipped_x).astype(np.int64)
    y0 = np.floor(clipped_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = clipped_x - x0
    dy = clipped_y - y0
    sampled = (
        image[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + image[y0, x1] * dx * (1.0 - dy)
        + image[y1, x0] * (1.0 - dx) * dy
        + image[y1, x1] * dx * dy
    )
    return sampled, valid


def aligned_depth_consistency(
    reference_depth: ArrayLike,
    candidate_depth: ArrayLike,
    reference_source_to_model: ArrayLike,
    candidate_source_to_model: ArrayLike,
    *,
    minimum_depth: float = 1e-8,
) -> dict[str, object]:
    """Compare depth on the common source-image support after scale alignment.

    Each affine maps the same pre-intervention source pixels into that run's
    model tensor. Candidate depth is bilinearly sampled at pixels corresponding
    to the reference tensor, then one robust scene-level scale is estimated by
    the median reference/candidate depth ratio.
    """

    reference = _depth_stack(reference_depth)
    candidate = _depth_stack(candidate_depth)
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate must have the same view count")
    reference_affines = _affine_stack(reference_source_to_model, len(reference))
    candidate_affines = _affine_stack(candidate_source_to_model, len(reference))

    correspondences: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    ratios: list[np.ndarray] = []
    for view, (reference_image, candidate_image) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        height, width = reference_image.shape
        grid_y, grid_x = np.mgrid[:height, :width]
        pixels = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.ones(height * width)],
            axis=0,
        )
        source_pixels = np.linalg.solve(reference_affines[view], pixels)
        source_pixels /= source_pixels[2:3]
        candidate_pixels = candidate_affines[view] @ source_pixels
        candidate_x = candidate_pixels[0] / candidate_pixels[2]
        candidate_y = candidate_pixels[1] / candidate_pixels[2]
        sampled_candidate, in_bounds = _bilinear_sample(
            candidate_image, candidate_x, candidate_y
        )
        flattened_reference = reference_image.reshape(-1)
        valid = (
            in_bounds
            & np.isfinite(flattened_reference)
            & np.isfinite(sampled_candidate)
            & (flattened_reference > minimum_depth)
            & (sampled_candidate > minimum_depth)
        )
        correspondences.append((flattened_reference, sampled_candidate, valid))
        ratios.append(flattened_reference[valid] / sampled_candidate[valid])

    nonempty_ratios = [item for item in ratios if len(item)]
    if not nonempty_ratios:
        raise ValueError("no valid depth correspondences remain after inverse mapping")
    scale = float(np.median(np.concatenate(nonempty_ratios)))
    per_view: list[dict[str, float | int]] = []
    all_errors: list[np.ndarray] = []
    for flattened_reference, sampled_candidate, valid in correspondences:
        errors = (
            np.abs(scale * sampled_candidate[valid] - flattened_reference[valid])
            / flattened_reference[valid]
        )
        all_errors.append(errors)
        per_view.append(
            {
                "valid_pixels": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "mean_abs_rel": float(np.mean(errors)) if len(errors) else float("nan"),
                "median_abs_rel": (
                    float(np.median(errors)) if len(errors) else float("nan")
                ),
            }
        )
    combined = np.concatenate(all_errors)
    return {
        "scale": scale,
        "valid_pixels": int(sum(len(errors) for errors in all_errors)),
        "mean_abs_rel": float(np.mean(combined)),
        "median_abs_rel": float(np.median(combined)),
        "p90_abs_rel": float(np.quantile(combined, 0.9)),
        "per_view": per_view,
    }
