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
