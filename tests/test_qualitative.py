import numpy as np

from camcanon3r.qualitative import (
    apply_camera_pose_alignment,
    median_camera_baseline,
    rasterize_aligned_points,
    source_supported_prediction_points,
)


def test_source_support_and_deterministic_cap() -> None:
    points = np.zeros((2, 2, 3, 3), dtype=np.float64)
    points[..., 2] = 2.0
    points[0, 0, 0] = np.nan
    affines = np.stack([np.eye(3), np.eye(3)])
    selected = source_supported_prediction_points(
        points,
        affines,
        [(3, 2), (3, 2)],
        maximum_per_view=2,
    )
    assert selected.shape == (4, 3)
    assert np.isfinite(selected).all()


def test_alignment_baseline_and_z_buffer_are_camera_defined() -> None:
    points = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [0.5, 0.0, 2.0]])
    alignment = {
        "scale": 2.0,
        "rotation": np.eye(3).tolist(),
        "translation": [0.0, 0.0, 0.0],
    }
    aligned = apply_camera_pose_alignment(points, alignment)
    np.testing.assert_allclose(aligned, points * 2.0)

    extrinsics = np.asarray(
        [
            np.column_stack([np.eye(3), [0.0, 0.0, 0.0]]),
            np.column_stack([np.eye(3), [-2.0, 0.0, 0.0]]),
        ]
    )
    assert median_camera_baseline(extrinsics) == 2.0
    intrinsic = np.asarray([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
    raster = rasterize_aligned_points(
        aligned,
        intrinsic,
        extrinsics[0],
        (4, 4),
        output_size=(4, 4),
        baseline=2.0,
    )
    assert raster.shape == (4, 4)
    assert np.count_nonzero(np.isfinite(raster)) == 2
    assert raster[2, 2] == np.log10(2.0)
