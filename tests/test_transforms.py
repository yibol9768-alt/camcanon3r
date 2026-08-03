import numpy as np
import pytest

from camcanon3r.metrics import (
    focal_relative_error,
    principal_point_error,
    rotation_geodesic_degrees,
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
