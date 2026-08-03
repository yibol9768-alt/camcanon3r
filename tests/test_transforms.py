import numpy as np
import pytest

from camcanon3r.metrics import (
    aligned_depth_consistency,
    focal_relative_error,
    pairwise_relative_pose_errors,
    principal_point_error,
    rotation_geodesic_degrees,
    translation_direction_degrees,
)
from camcanon3r.transforms import crop_resize, letterbox, resize


def test_crop_resize_updates_focal_and_principal_point_exactly() -> None:
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]])
    transform = crop_resize((640, 480), (80.0, 40.0, 400.0, 300.0), (800, 600))
    transformed = transform.transform_intrinsics(intrinsics)
    expected = np.array([[1600.0, 0.0, 480.0], [0.0, 1620.0, 400.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(transform.inverse_intrinsics(transformed), intrinsics)


def test_letterbox_moves_principal_point_with_padding() -> None:
    intrinsics = np.array([[1000.0, 0.0, 400.0], [0.0, 1000.0, 200.0], [0.0, 0.0, 1.0]])
    transform = letterbox((800, 400), (800, 800))
    transformed = transform.transform_intrinsics(intrinsics)
    np.testing.assert_allclose(transformed[:2, 2], [400.0, 400.0])


def test_composition_matches_sequential_pixel_mapping() -> None:
    first = crop_resize((1000, 800), (100.0, 80.0, 800.0, 640.0), (500, 400))
    second = resize((500, 400), (750, 500))
    combined = first.compose(second)
    pixels = np.array([[100.0, 80.0], [900.0, 720.0], [500.0, 400.0]])
    np.testing.assert_allclose(
        combined.transform_pixels(pixels),
        second.transform_pixels(first.transform_pixels(pixels)),
    )


def test_invalid_crop_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        crop_resize((640, 480), (600.0, 0.0, 100.0, 100.0), (224, 224))


def test_camera_metrics_have_expected_zero_and_known_rotation() -> None:
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    assert focal_relative_error(intrinsics, intrinsics) == 0.0
    assert principal_point_error(intrinsics, intrinsics, (640, 480)) == 0.0
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert rotation_geodesic_degrees(rotation, np.eye(3)) == pytest.approx(30.0)
    assert translation_direction_degrees([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)


def test_pairwise_pose_errors_cancel_global_similarity_gauge() -> None:
    reference = np.array(
        [
            np.hstack([np.eye(3), np.array([[0.0], [0.0], [0.0]])]),
            np.hstack([np.eye(3), np.array([[-1.0], [0.0], [0.0]])]),
            np.hstack([np.eye(3), np.array([[0.0], [-2.0], [0.0]])]),
        ]
    )
    angle = np.deg2rad(25.0)
    gauge_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 3.5
    world_translation = np.array([4.0, -2.0, 1.0])
    reference_rotations = reference[:, :, :3]
    reference_translations = reference[:, :, 3]
    centers = -np.einsum("vji,vj->vi", reference_rotations, reference_translations)
    transformed_centers = scale * (centers @ gauge_rotation.T) + world_translation
    transformed_rotations = reference_rotations @ gauge_rotation.T
    transformed_translations = -np.einsum(
        "vij,vj->vi", transformed_rotations, transformed_centers
    )
    candidate = np.concatenate(
        [transformed_rotations, transformed_translations[:, :, None]], axis=2
    )

    errors = pairwise_relative_pose_errors(reference, candidate)
    np.testing.assert_allclose(errors["rotation_degrees"], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        errors["translation_direction_degrees"], 0.0, atol=1e-6
    )


def test_aligned_depth_consistency_recovers_scale_and_common_crop() -> None:
    reference = np.stack(
        [
            np.tile(np.arange(1.0, 5.0), (3, 1)),
            np.tile(np.arange(2.0, 6.0), (3, 1)),
        ]
    )
    candidate = np.zeros_like(reference)
    candidate[:, :, :3] = reference[:, :, 1:] / 2.5
    identity = np.repeat(np.eye(3)[None], 2, axis=0)
    shifted = identity.copy()
    shifted[:, 0, 2] = -1.0

    result = aligned_depth_consistency(
        reference,
        candidate,
        identity,
        shifted,
    )
    assert result["scale"] == pytest.approx(2.5)
    assert result["valid_pixels"] == 18
    assert result["mean_abs_rel"] == pytest.approx(0.0)
    assert all(view["valid_fraction"] == pytest.approx(0.75) for view in result["per_view"])
