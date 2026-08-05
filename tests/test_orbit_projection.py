import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from camcanon3r.metrics import (
    camera_centers_from_extrinsics,
    pairwise_relative_pose_errors,
)
from camcanon3r.orbit_projection import (
    chordal_rotation_mean,
    orbit_medoid,
    project_camera_orbit,
    project_camera_response_field,
    relative_rotation_graph,
    synchronize_rotations,
)

MEMBERS = (
    "center",
    "left",
    "right",
    "top",
    "bottom",
    "top_left",
    "bottom_right",
    "top_right",
    "bottom_left",
)
INVERSES = {
    "center": "center",
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
    "top_left": "bottom_right",
    "bottom_right": "top_left",
    "top_right": "bottom_left",
    "bottom_left": "top_right",
}
PLACEMENTS = {
    "center": (0.5, 0.5),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "top": (0.5, 0.0),
    "bottom": (0.5, 1.0),
    "top_left": (0.0, 0.0),
    "bottom_right": (1.0, 1.0),
    "top_right": (1.0, 0.0),
    "bottom_left": (0.0, 1.0),
}


def _extrinsics(rotvecs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_rotvec(rotvecs).as_matrix()
    translations = -np.einsum("vij,vj->vi", rotations, centers)
    return np.concatenate([rotations, translations[:, :, None]], axis=2)


def _canonical() -> np.ndarray:
    return _extrinsics(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.08, -0.03, 0.02],
                [-0.05, 0.10, 0.01],
                [0.04, 0.06, -0.09],
            ]
        ),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.2],
                [0.3, 1.4, -0.1],
                [-0.4, 0.5, 1.1],
            ]
        ),
    )


def _change_gauge(
    extrinsics: np.ndarray,
    *,
    rotation_vector: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> np.ndarray:
    world_rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    rotations = extrinsics[:, :, :3] @ world_rotation
    centers = camera_centers_from_extrinsics(extrinsics)
    changed_centers = (world_rotation.T @ (centers - translation).T).T / scale
    changed_translations = -np.einsum("vij,vj->vi", rotations, changed_centers)
    return np.concatenate([rotations, changed_translations[:, :, None]], axis=2)


def _gauged_members(extrinsics: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for index, label in enumerate(MEMBERS):
        result[label] = _change_gauge(
            extrinsics,
            rotation_vector=np.asarray(
                [0.03 * index, -0.02 * index, 0.01 * (index + 1)]
            ),
            scale=0.8 + 0.1 * index,
            translation=np.asarray([0.2 * index, -0.1 * index, 0.05 * index]),
        )
    return result


def _median_rotation_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.median(
            pairwise_relative_pose_errors(reference, candidate)["rotation_degrees"]
        )
    )


def _view_biased(extrinsics: np.ndarray, angle_degrees: float) -> np.ndarray:
    axes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
            [-0.3, 1.0, 0.1],
            [0.2, -0.4, 1.0],
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes[1:] /= norms[1:]
    perturbations = Rotation.from_rotvec(np.radians(angle_degrees) * axes).as_matrix()
    rotations = perturbations @ extrinsics[:, :, :3]
    centers = camera_centers_from_extrinsics(extrinsics)
    translations = -np.einsum("vij,vj->vi", rotations, centers)
    return np.concatenate([rotations, translations[:, :, None]], axis=2)


def test_relative_rotation_graph_is_world_gauge_invariant():
    canonical = _canonical()
    changed = _change_gauge(
        canonical,
        rotation_vector=np.asarray([0.3, -0.2, 0.1]),
        scale=2.7,
        translation=np.asarray([4.0, -1.0, 0.5]),
    )

    assert np.allclose(
        relative_rotation_graph(canonical),
        relative_rotation_graph(changed),
        atol=1e-12,
    )


def test_chordal_mean_is_a_proper_rotation():
    rotations = Rotation.from_rotvec(
        [[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3]]
    ).as_matrix()
    mean = chordal_rotation_mean(rotations, weights=[1.0, 2.0, 3.0])

    assert np.allclose(mean.T @ mean, np.eye(3), atol=1e-12)
    assert np.linalg.det(mean) == pytest.approx(1.0)


def test_synchronization_closes_a_noisy_complete_graph():
    graph = relative_rotation_graph(_canonical())
    noisy = graph.copy()
    perturbation = Rotation.from_rotvec([0.01, -0.02, 0.005]).as_matrix()
    noisy[1, 3] = perturbation @ noisy[1, 3]
    noisy[3, 1] = noisy[1, 3].T

    rotations, diagnostics = synchronize_rotations(noisy)
    synchronized = _extrinsics(
        Rotation.from_matrix(rotations).as_rotvec(),
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    )

    assert (
        diagnostics["weighted_rms_radians_after"]
        < diagnostics["weighted_rms_radians_before"]
    )
    assert np.allclose(relative_rotation_graph(synchronized)[0, 0], np.eye(3))


def test_consistent_orbit_projects_exactly_modulo_gauge_and_scale():
    canonical = _canonical()
    result = project_camera_orbit(
        _gauged_members(canonical),
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
    )

    errors = pairwise_relative_pose_errors(canonical, result["extrinsic"])
    assert np.max(errors["rotation_degrees"]) < 1e-5
    assert np.nanmax(errors["translation_direction_degrees"]) < 1e-5
    assert result["ground_truth_used"] is False
    assert result["native_confidence_used"] is False


def test_projection_is_independent_of_mapping_insertion_order():
    members = _gauged_members(_canonical())
    forward = project_camera_orbit(
        members,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
    )
    reverse_mapping = {label: members[label] for label in reversed(MEMBERS)}
    reverse = project_camera_orbit(
        reverse_mapping,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
    )

    assert np.allclose(forward["extrinsic"], reverse["extrinsic"], atol=1e-12)
    assert forward["member_weights"] == reverse["member_weights"]


def test_symmetric_pairing_and_robust_weight_reject_one_bad_group():
    canonical = _canonical()
    signed_bias = {
        "center": 0.2,
        "left": 1.5,
        "right": -1.5,
        "top": 2.0,
        "bottom": -2.0,
        "top_left": 2.5,
        "bottom_right": -2.5,
        "top_right": 22.0,
        "bottom_left": 22.0,
    }
    members = _gauged_members(canonical)
    members = {
        label: _view_biased(value, signed_bias[label])
        for label, value in members.items()
    }

    robust = project_camera_orbit(
        members,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
        robust=True,
    )
    uniform = project_camera_orbit(
        members,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
        robust=False,
    )

    robust_error = _median_rotation_error(canonical, robust["extrinsic"])
    uniform_error = _median_rotation_error(canonical, uniform["extrinsic"])
    assert robust_error < 0.5
    assert robust_error < uniform_error
    assert robust["group_weights"]["top_right+bottom_left"] == pytest.approx(0.0)


def test_orbit_medoid_is_deterministic_for_exact_ties():
    members = _gauged_members(_canonical())
    selected, scores = orbit_medoid(members, member_order=MEMBERS)

    assert selected == "center"
    assert set(scores) == set(MEMBERS)


def test_projection_rejects_noninvolutive_inverse_pairs():
    inverse = dict(INVERSES)
    inverse["left"] = "top"
    with pytest.raises(ValueError, match="involution"):
        project_camera_orbit(
            _gauged_members(_canonical()),
            member_order=MEMBERS,
            inverse_pairs=inverse,
        )


def test_projection_keeps_rotation_when_every_camera_center_is_degenerate():
    rotations = _canonical()[:, :, :3]
    poses = np.concatenate([rotations, np.zeros((4, 3, 1))], axis=2)
    projected = project_camera_orbit(
        {label: poses for label in MEMBERS},
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
    )

    assert projected["translation_status"] == (
        "undefined_all_member_centers_degenerate"
    )
    assert projected["translation_member_labels"] == []
    assert np.allclose(projected["extrinsic"][:, :, 3], 0.0)


def test_response_field_selects_constant_for_a_consistent_orbit():
    canonical = _canonical()
    projected = project_camera_response_field(
        _gauged_members(canonical),
        placements=PLACEMENTS,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
    )

    assert projected["selected_basis"] == "constant"
    assert _median_rotation_error(canonical, projected["extrinsic"]) < 1e-5
    assert projected["ground_truth_used"] is False


def test_quadratic_response_field_removes_even_canvas_bias():
    canonical = _canonical()
    members = _gauged_members(canonical)
    biased = {}
    for label, value in members.items():
        x = 2.0 * PLACEMENTS[label][0] - 1.0
        y = 2.0 * PLACEMENTS[label][1] - 1.0
        angle = 4.0 * x * x + 2.0 * y * y + 1.5 * x - y
        biased[label] = _view_biased(value, angle)

    response = project_camera_response_field(
        biased,
        placements=PLACEMENTS,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
    )
    robust_group = project_camera_orbit(
        biased,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
    )

    response_error = _median_rotation_error(canonical, response["extrinsic"])
    group_error = _median_rotation_error(canonical, robust_group["extrinsic"])
    assert response["selected_basis"] == "quadratic"
    assert response_error < 0.2
    assert response_error < 0.25 * group_error


def test_response_field_falls_back_when_intercept_leaves_center_trust_region():
    canonical = _canonical()
    members = _gauged_members(canonical)
    biases = {
        "center": 0.2,
        "left": 1.0,
        "right": -1.0,
        "top": 1.5,
        "bottom": -1.5,
        "top_left": 2.0,
        "bottom_right": -2.0,
        "top_right": 20.0,
        "bottom_left": 20.0,
    }
    biased = {
        label: _view_biased(value, biases[label]) for label, value in members.items()
    }
    response = project_camera_response_field(
        biased,
        placements=PLACEMENTS,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
        maximum_anchor_deviation_degrees=0.5,
    )
    robust_group = project_camera_orbit(
        biased,
        member_order=MEMBERS,
        inverse_pairs=INVERSES,
    )

    assert response["response_fallback_used"] is True
    assert response["response_fallback"] == "inverse_pair_robust_group_projection"
    assert response["geometry_member_weights"] == robust_group["member_weights"]
    assert np.allclose(response["extrinsic"], robust_group["extrinsic"])
