"""Camera-space metrics used by the CamCanon3R protocol."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


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
    predicted_r = np.asarray(predicted, dtype=np.float64)
    target_r = np.asarray(target, dtype=np.float64)
    relative = predicted_r @ target_r.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
