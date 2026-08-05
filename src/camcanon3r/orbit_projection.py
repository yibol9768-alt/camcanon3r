"""Gauge-invariant projection of a camera-prediction orbit.

The module treats each reconstruction as a camera graph modulo an arbitrary
world Sim(3). Relative camera rotations and root-camera center coordinates are
the quotient observables. A symmetric placement orbit is reduced to inverse
pairs, robustly averaged, and synchronized into one new camera reconstruction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .metrics import camera_centers_from_extrinsics, rotation_geodesic_degrees


def _project_so3(matrix: ArrayLike) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation candidate must be finite with shape 3x3")
    left, _, right_transposed = np.linalg.svd(value)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_transposed) < 0.0:
        signs[-1] = -1.0
    return left @ np.diag(signs) @ right_transposed


def _validate_extrinsics(value: ArrayLike) -> np.ndarray:
    extrinsics = np.asarray(value, dtype=np.float64)
    while extrinsics.ndim > 3 and extrinsics.shape[0] == 1:
        extrinsics = extrinsics[0]
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(
            "orbit extrinsics must have shape (views, 3, 4) or (views, 4, 4)"
        )
    extrinsics = extrinsics[:, :3, :4].copy()
    if len(extrinsics) < 2 or not np.isfinite(extrinsics).all():
        raise ValueError("orbit extrinsics require at least two finite views")
    for index, rotation in enumerate(extrinsics[:, :, :3]):
        projected = _project_so3(rotation)
        if not np.allclose(rotation, projected, atol=1e-4, rtol=1e-4):
            raise ValueError(f"view {index} is not a proper rotation")
        extrinsics[index, :, :3] = projected
    return extrinsics


def chordal_rotation_mean(
    rotations: ArrayLike, weights: ArrayLike | None = None
) -> np.ndarray:
    """Return the proper SO(3) projection of a weighted matrix mean."""

    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not len(values):
        raise ValueError("rotations must have shape (N, 3, 3)")
    if not np.isfinite(values).all():
        raise ValueError("rotations contain a non-finite value")
    if weights is None:
        normalized = np.full(len(values), 1.0 / len(values), dtype=np.float64)
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        if normalized.shape != (len(values),):
            raise ValueError("rotation weights must have shape (N,)")
        if not np.isfinite(normalized).all() or np.any(normalized < 0.0):
            raise ValueError("rotation weights must be finite and non-negative")
        total = float(np.sum(normalized))
        if total <= 0.0:
            raise ValueError("at least one rotation weight must be positive")
        normalized = normalized / total
    return _project_so3(np.einsum("n,nij->ij", normalized, values))


def relative_rotation_graph(extrinsics: ArrayLike) -> np.ndarray:
    """Return gauge-invariant rotations ``R_j R_i^T`` for every directed edge."""

    poses = _validate_extrinsics(extrinsics)
    return _relative_rotation_graph(poses)


def _relative_rotation_graph(extrinsics: np.ndarray) -> np.ndarray:
    rotations = extrinsics[:, :, :3]
    view_count = len(rotations)
    graph = np.empty((view_count, view_count, 3, 3), dtype=np.float64)
    for first in range(view_count):
        for second in range(view_count):
            graph[first, second] = rotations[second] @ rotations[first].T
    return graph


def _symmetry_groups(
    member_order: Sequence[str], inverse_pairs: Mapping[str, str]
) -> list[tuple[str, ...]]:
    labels = [str(label) for label in member_order]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("member order must be non-empty and unique")
    if set(inverse_pairs) != set(labels):
        raise ValueError("inverse-pair mapping must cover every orbit member")
    order_index = {label: index for index, label in enumerate(labels)}
    groups: list[tuple[str, ...]] = []
    visited: set[str] = set()
    for label in labels:
        if label in visited:
            continue
        partner = str(inverse_pairs[label])
        if partner not in order_index or str(inverse_pairs.get(partner)) != label:
            raise ValueError("inverse-pair mapping must be an involution")
        group = (label,) if partner == label else (label, partner)
        visited.update(group)
        groups.append(group)
    return groups


def _edge_indices(view_count: int) -> list[tuple[int, int]]:
    return [
        (first, second)
        for first in range(view_count - 1)
        for second in range(first + 1, view_count)
    ]


def _edge_residuals(
    graphs: np.ndarray, target: np.ndarray, edges: Sequence[tuple[int, int]]
) -> np.ndarray:
    residuals = np.empty(len(graphs), dtype=np.float64)
    for graph_index, graph in enumerate(graphs):
        values = [
            rotation_geodesic_degrees(graph[first, second], target[first, second])
            for first, second in edges
        ]
        residuals[graph_index] = float(np.median(values))
    return residuals


def _tukey_weights(
    residuals_degrees: np.ndarray,
    *,
    tuning_constant: float,
    scale_floor_degrees: float,
) -> tuple[np.ndarray, float]:
    if tuning_constant <= 0.0 or scale_floor_degrees <= 0.0:
        raise ValueError("Tukey constants must be positive")
    residuals = np.asarray(residuals_degrees, dtype=np.float64)
    if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("Tukey residuals must be a finite vector")
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_scale = max(1.4826 * mad, median / 0.6745, scale_floor_degrees)
    normalized = residuals / (tuning_constant * robust_scale)
    weights = np.square(1.0 - np.square(normalized))
    weights[normalized >= 1.0] = 0.0
    return weights, robust_scale


def _weighted_edge_graph(
    graphs: np.ndarray,
    weights: np.ndarray,
    *,
    view_count: int,
) -> np.ndarray:
    target = np.empty((view_count, view_count, 3, 3), dtype=np.float64)
    for first in range(view_count):
        target[first, first] = np.eye(3)
        for second in range(first + 1, view_count):
            mean = chordal_rotation_mean(graphs[:, first, second], weights=weights)
            target[first, second] = mean
            target[second, first] = mean.T
    return target


def synchronize_rotations(
    relative_graph: ArrayLike,
    *,
    edge_weights: Mapping[tuple[int, int], float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Recover a cycle-consistent camera rotation set with camera zero as gauge."""

    graph = np.asarray(relative_graph, dtype=np.float64)
    if (
        graph.ndim != 4
        or graph.shape[0] != graph.shape[1]
        or graph.shape[2:] != (3, 3)
        or graph.shape[0] < 2
        or not np.isfinite(graph).all()
    ):
        raise ValueError("relative graph must have shape (V, V, 3, 3)")
    view_count = graph.shape[0]
    edges = _edge_indices(view_count)
    weights = {
        edge: float(edge_weights[edge]) if edge_weights is not None else 1.0
        for edge in edges
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise ValueError("synchronization edge weights must be finite and positive")

    initial = np.stack(
        [np.eye(3)] + [_project_so3(graph[0, view]) for view in range(1, view_count)]
    )
    initial_parameters = Rotation.from_matrix(initial[1:]).as_rotvec().reshape(-1)

    def unpack(parameters: np.ndarray) -> np.ndarray:
        rotations = np.empty((view_count, 3, 3), dtype=np.float64)
        rotations[0] = np.eye(3)
        rotations[1:] = Rotation.from_rotvec(
            parameters.reshape(view_count - 1, 3)
        ).as_matrix()
        return rotations

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotations = unpack(parameters)
        values: list[np.ndarray] = []
        for first, second in edges:
            predicted = rotations[second] @ rotations[first].T
            error = graph[first, second].T @ predicted
            values.append(
                np.sqrt(weights[(first, second)])
                * Rotation.from_matrix(error).as_rotvec()
            )
        return np.concatenate(values)

    before = residual(initial_parameters)
    optimized = least_squares(
        residual,
        initial_parameters,
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise RuntimeError(f"SO(3) synchronization failed: {optimized.message}")
    rotations = unpack(optimized.x)
    after = residual(optimized.x)
    diagnostics = {
        "weighted_rms_radians_before": float(np.sqrt(np.mean(before**2))),
        "weighted_rms_radians_after": float(np.sqrt(np.mean(after**2))),
        "optimization_cost": float(optimized.cost),
        "optimization_evaluations": int(optimized.nfev),
    }
    return rotations, diagnostics


def _normalized_root_centers(extrinsics: np.ndarray) -> np.ndarray:
    centers = camera_centers_from_extrinsics(extrinsics)
    root = extrinsics[0, :, :3]
    rooted = (root @ (centers - centers[0]).T).T
    pairwise = np.linalg.norm(rooted[:, None, :] - rooted[None, :, :], axis=-1)
    nonzero = pairwise[np.triu_indices(len(rooted), k=1)]
    nonzero = nonzero[nonzero > 1e-12]
    if not len(nonzero):
        raise ValueError("camera centers are degenerate for orbit projection")
    return rooted / float(np.median(nonzero))


def _weighted_geometric_median(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    tolerance: float = 1e-10,
    maximum_iterations: int = 500,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    normalized = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("geometric-median points must have shape (N, 3)")
    if normalized.shape != (len(values),) or np.sum(normalized) <= 0.0:
        raise ValueError("geometric-median weights are invalid")
    normalized = normalized / np.sum(normalized)
    current = np.einsum("n,ni->i", normalized, values)
    for _ in range(maximum_iterations):
        distances = np.linalg.norm(values - current, axis=1)
        coincident = np.flatnonzero(distances <= tolerance)
        noncoincident = distances > tolerance
        if not np.any(noncoincident):
            return current
        inverse = normalized[noncoincident] / distances[noncoincident]
        weiszfeld = np.einsum("n,ni->i", inverse, values[noncoincident]) / np.sum(
            inverse
        )
        coincident_weight = float(np.sum(normalized[coincident]))
        if coincident_weight:
            residual = np.einsum(
                "n,ni->i",
                inverse,
                values[noncoincident] - current,
            )
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= coincident_weight:
                return current
            ratio = coincident_weight / residual_norm
            candidate = (1.0 - ratio) * weiszfeld + ratio * current
        else:
            candidate = weiszfeld
        if np.linalg.norm(candidate - current) <= tolerance:
            return candidate
        current = candidate
    raise RuntimeError("weighted geometric median did not converge")


def orbit_medoid(
    members: Mapping[str, ArrayLike], *, member_order: Sequence[str]
) -> tuple[str, dict[str, float]]:
    """Select the member with minimum median camera-graph disagreement."""

    labels = [str(label) for label in member_order]
    if set(members) != set(labels):
        raise ValueError("orbit medoid members do not match member order")
    extrinsics = {label: _validate_extrinsics(members[label]) for label in labels}
    view_counts = {len(value) for value in extrinsics.values()}
    if len(view_counts) != 1:
        raise ValueError("all orbit members must have the same view count")
    edges = _edge_indices(next(iter(view_counts)))
    graphs = {
        label: _relative_rotation_graph(value) for label, value in extrinsics.items()
    }
    scores: dict[str, float] = {}
    for label in labels:
        disagreements = []
        for other in labels:
            if other == label:
                continue
            disagreements.append(
                np.median(
                    [
                        rotation_geodesic_degrees(
                            graphs[label][first, second],
                            graphs[other][first, second],
                        )
                        for first, second in edges
                    ]
                )
            )
        scores[label] = float(np.median(disagreements))
    order_index = {label: index for index, label in enumerate(labels)}
    selected = min(labels, key=lambda label: (scores[label], order_index[label]))
    return selected, scores


def project_camera_orbit(
    members: Mapping[str, ArrayLike],
    *,
    member_order: Sequence[str],
    inverse_pairs: Mapping[str, str],
    robust: bool = True,
    tuning_constant: float = 4.685,
    scale_floor_degrees: float = 0.25,
    minimum_effective_groups: int = 3,
) -> dict[str, Any]:
    """Project an orbit of camera predictions into one quotient consensus."""

    labels = [str(label) for label in member_order]
    if set(members) != set(labels):
        raise ValueError("orbit members do not match the frozen member order")
    poses = {label: _validate_extrinsics(members[label]) for label in labels}
    view_counts = {len(value) for value in poses.values()}
    if len(view_counts) != 1:
        raise ValueError("all orbit members must have the same view count")
    view_count = next(iter(view_counts))
    edges = _edge_indices(view_count)
    groups = _symmetry_groups(labels, inverse_pairs)

    graphs = {label: _relative_rotation_graph(poses[label]) for label in labels}
    group_graphs = []
    for group in groups:
        graph = np.empty((view_count, view_count, 3, 3), dtype=np.float64)
        for first in range(view_count):
            graph[first, first] = np.eye(3)
            for second in range(first + 1, view_count):
                mean = chordal_rotation_mean(
                    np.stack([graphs[label][first, second] for label in group])
                )
                graph[first, second] = mean
                graph[second, first] = mean.T
        group_graphs.append(graph)
    group_graph_array = np.stack(group_graphs)
    group_weights = np.ones(len(groups), dtype=np.float64)
    robust_scale = None
    for _ in range(20 if robust else 1):
        target = _weighted_edge_graph(
            group_graph_array, group_weights, view_count=view_count
        )
        residuals = _edge_residuals(group_graph_array, target, edges)
        if not robust:
            break
        updated, robust_scale = _tukey_weights(
            residuals,
            tuning_constant=tuning_constant,
            scale_floor_degrees=scale_floor_degrees,
        )
        if np.count_nonzero(updated > 1e-8) < minimum_effective_groups:
            updated = np.ones(len(groups), dtype=np.float64)
        if np.allclose(
            updated / np.sum(updated), group_weights / np.sum(group_weights)
        ):
            group_weights = updated
            break
        group_weights = updated
    target = _weighted_edge_graph(
        group_graph_array, group_weights, view_count=view_count
    )
    residuals = _edge_residuals(group_graph_array, target, edges)
    rotations, synchronization = synchronize_rotations(target)

    member_weights: dict[str, float] = {}
    for group, weight in zip(groups, group_weights, strict=True):
        for label in group:
            member_weights[label] = float(weight / len(group))
    center_labels: list[str] = []
    rooted_center_records: list[np.ndarray] = []
    for label in labels:
        try:
            rooted = _normalized_root_centers(poses[label])
        except ValueError as error:
            if "degenerate" not in str(error):
                raise
            continue
        center_labels.append(label)
        rooted_center_records.append(rooted)
    if rooted_center_records:
        rooted_centers = np.stack(rooted_center_records)
        center_weights = np.asarray([member_weights[label] for label in center_labels])
        centers = np.stack(
            [
                _weighted_geometric_median(rooted_centers[:, view], center_weights)
                for view in range(view_count)
            ]
        )
        centers[0] = 0.0
        translation_status = "available"
    else:
        centers = np.zeros((view_count, 3), dtype=np.float64)
        translation_status = "undefined_all_member_centers_degenerate"
    translations = -np.einsum("vij,vj->vi", rotations, centers)
    projected = np.concatenate([rotations, translations[:, :, None]], axis=2)
    selected_medoid, medoid_scores = orbit_medoid(members, member_order=labels)

    return {
        "extrinsic": projected,
        "rotation": rotations,
        "camera_center": centers,
        "translation_status": translation_status,
        "translation_member_labels": center_labels,
        "member_order": labels,
        "symmetry_groups": [list(group) for group in groups],
        "group_weights": {
            "+".join(group): float(weight)
            for group, weight in zip(groups, group_weights, strict=True)
        },
        "group_residual_degrees": {
            "+".join(group): float(residual)
            for group, residual in zip(groups, residuals, strict=True)
        },
        "member_weights": member_weights,
        "robust_scale_degrees": (
            float(robust_scale) if robust_scale is not None else None
        ),
        "synchronization": synchronization,
        "orbit_medoid": selected_medoid,
        "orbit_medoid_scores_degrees": medoid_scores,
        "ground_truth_used": False,
        "native_confidence_used": False,
    }
