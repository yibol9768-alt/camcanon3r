import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from camcanon3r.metrics import camera_centers_from_extrinsics
from camcanon3r.orbit_fusion import (
    apply_world_similarity,
    camera_world_similarity,
    fuse_orbit_geometry,
    fuse_source_intrinsics,
)


def _extrinsics(rotvecs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_rotvec(rotvecs).as_matrix()
    translations = -np.einsum("vij,vj->vi", rotations, centers)
    return np.concatenate([rotations, translations[:, :, None]], axis=2)


def _target_cameras() -> np.ndarray:
    return _extrinsics(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.08, -0.03, 0.02],
                [-0.05, 0.10, 0.01],
            ]
        ),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.2],
                [0.3, 1.4, -0.1],
            ]
        ),
    )


def _inverse_gauge(
    target: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    target_rotations = target[:, :, :3]
    source_rotations = np.einsum("vij,jk->vik", target_rotations, rotation)
    target_centers = camera_centers_from_extrinsics(target)
    source_centers = (rotation.T @ ((target_centers - translation) / scale).T).T
    source_translations = -np.einsum("vij,vj->vi", source_rotations, source_centers)
    return np.concatenate([source_rotations, source_translations[:, :, None]], axis=2)


def test_camera_world_similarity_recovers_exact_sim3():
    target = _target_cameras()
    rotation = Rotation.from_rotvec([0.2, -0.1, 0.05]).as_matrix()
    source = _inverse_gauge(
        target,
        scale=1.7,
        rotation=rotation,
        translation=np.asarray([0.4, -0.2, 0.7]),
    )

    similarity = camera_world_similarity(source, target)

    assert similarity["scale"] == pytest.approx(1.7)
    np.testing.assert_allclose(similarity["rotation"], rotation, atol=1e-12)
    np.testing.assert_allclose(similarity["translation"], [0.4, -0.2, 0.7], atol=1e-12)
    assert similarity["maximum_rotation_residual_degrees"] < 1e-5
    assert similarity["maximum_center_residual_normalized"] < 1e-12

    points = np.asarray([[0.1, 0.2, 0.3], [-1.0, 2.0, 0.5]])
    expected = 1.7 * (rotation @ points.T).T + [0.4, -0.2, 0.7]
    np.testing.assert_allclose(
        apply_world_similarity(points, similarity), expected, atol=1e-12
    )


def test_fuse_source_intrinsics_removes_registered_pixel_affines():
    source = np.asarray(
        [
            [[500.0, 0.0, 100.0], [0.0, 510.0, 80.0], [0.0, 0.0, 1.0]],
            [[520.0, 0.0, 110.0], [0.0, 530.0, 90.0], [0.0, 0.0, 1.0]],
        ]
    )
    affines = np.asarray(
        [
            [
                [[0.5, 0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 0.0, 1.0]],
                [[0.5, 0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[0.5, 0.0, 0.0], [0.0, 0.5, 2.0], [0.0, 0.0, 1.0]],
                [[0.5, 0.0, 0.0], [0.0, 0.5, 2.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[0.5, 0.0, 2.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
                [[0.5, 0.0, 2.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            ],
        ]
    )
    model = np.einsum("mvij,vjk->mvik", affines, source)

    fused_source, fused_model = fuse_source_intrinsics(
        model,
        affines,
        member_weights=[1.0, 1.0, 1.0],
        reference_index=0,
    )

    np.testing.assert_allclose(fused_source, source, atol=1e-12)
    np.testing.assert_allclose(
        fused_model, np.einsum("vij,vjk->vik", affines[0], source), atol=1e-12
    )


def _member_prediction(
    target_extrinsics: np.ndarray,
    affine: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    outlier: bool = False,
) -> dict[str, np.ndarray]:
    view_count = len(target_extrinsics)
    source_extrinsics = _inverse_gauge(
        target_extrinsics,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    grid_y, grid_x = np.mgrid[:5, :5]
    model_pixels = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), np.ones(25)])
    world_points = []
    for view in range(view_count):
        source_pixels = np.linalg.solve(affine, model_pixels)
        source_x = source_pixels[0] / source_pixels[2]
        source_y = source_pixels[1] / source_pixels[2]
        target_points = np.column_stack([source_x, source_y, np.full(25, 2.0 + view)])
        if outlier:
            target_points += np.asarray([20.0, -10.0, 5.0])
        source_points = (rotation.T @ ((target_points - translation) / scale).T).T
        world_points.append(source_points.reshape(5, 5, 3))
    source_intrinsic = np.repeat(
        np.asarray([[[4.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]]),
        view_count,
        axis=0,
    )
    affine_stack = np.repeat(affine[None], view_count, axis=0)
    return {
        "extrinsic": source_extrinsics,
        "intrinsic": np.einsum("vij,vjk->vik", affine_stack, source_intrinsic),
        "world_points": np.stack(world_points),
        "world_points_conf": np.ones((view_count, 5, 5)),
        "model_preprocess_affine": np.repeat(np.eye(3)[None], view_count, axis=0),
        "protocol_affine": affine_stack,
        "source_to_model_affine": affine_stack,
    }


def test_geometry_fusion_aligns_gauges_and_rejects_a_point_outlier():
    target = _target_cameras()
    labels = ("center", "left", "right", "bad")
    affines = {
        "center": np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        "left": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        "right": np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        "bad": np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    }
    members = {}
    for index, label in enumerate(labels):
        members[label] = _member_prediction(
            target,
            affines[label],
            scale=0.8 + 0.2 * index,
            rotation=Rotation.from_rotvec(
                [0.04 * index, -0.02 * index, 0.01 * index]
            ).as_matrix(),
            translation=np.asarray([0.1 * index, -0.2 * index, 0.05 * index]),
            outlier=label == "bad",
        )
    masks = [np.ones((3, 3), dtype=np.uint8) for _ in range(len(target))]

    fused = fuse_orbit_geometry(
        members,
        projected_extrinsics=target,
        member_order=labels,
        member_weights={label: 1.0 for label in labels},
        source_support_masks=masks,
        reference_label="center",
        minimum_members=3,
        tile_rows=2,
    )

    expected = np.empty((len(target), 3, 3, 3), dtype=np.float64)
    for view in range(len(target)):
        y, x = np.mgrid[:3, :3]
        expected[view] = np.stack([x, y, np.full((3, 3), 2.0 + view)], axis=-1)
    np.testing.assert_allclose(fused["world_points"][:, 1:4, 1:4], expected, atol=1e-5)
    assert np.isnan(fused["world_points"][:, 0]).all()
    assert fused["valid_fused_pixels"] == len(target) * 9
    assert fused["valid_fused_fraction"] == pytest.approx(9 / 25)
    assert fused["median_point_dispersion"] < 1e-5
    assert fused["ground_truth_used"] is False
    np.testing.assert_allclose(fused["extrinsic"], target)
    np.testing.assert_allclose(
        fused["source_intrinsic"][:, [0, 1, 0, 1], [0, 1, 2, 2]],
        np.asarray([[4.0, 4.0, 1.0, 1.0]] * len(target)),
    )
