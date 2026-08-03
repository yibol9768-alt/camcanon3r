import numpy as np
import pytest

from camcanon3r.dust3r_preprocess import plan_dust3r_preprocessing


def test_512_preprocessing_preserves_landscape_and_portrait_geometry() -> None:
    landscape, portrait = plan_dust3r_preprocessing(
        [(800, 400), (400, 800)]
    )
    assert landscape.resized_size == (512, 256)
    assert landscape.affine.target_size == (512, 256)
    assert portrait.resized_size == (256, 512)
    assert portrait.affine.target_size == (256, 512)
    np.testing.assert_allclose(
        landscape.affine.matrix,
        [[512 / 800, 0, 0], [0, 256 / 400, 0], [0, 0, 1]],
    )


def test_512_preprocessing_records_square_to_four_by_three_crop() -> None:
    (cropped,) = plan_dust3r_preprocessing([(400, 400)], square_ok=False)
    (square,) = plan_dust3r_preprocessing([(400, 400)], square_ok=True)
    assert cropped.crop_left_top_right_bottom == (0, 64, 512, 448)
    assert cropped.affine.target_size == (512, 384)
    np.testing.assert_allclose(
        cropped.affine.matrix,
        [[512 / 400, 0, 0], [0, 512 / 400, -64], [0, 0, 1]],
    )
    assert square.affine.target_size == (512, 512)


def test_512_preprocessing_retains_rounding_and_patch_crop() -> None:
    (spec,) = plan_dust3r_preprocessing([(1000, 667)])
    assert spec.resized_size == (512, 342)
    assert spec.crop_left_top_right_bottom == (0, 3, 512, 339)
    assert spec.affine.target_size == (512, 336)
    np.testing.assert_allclose(
        spec.affine.matrix,
        [[512 / 1000, 0, 0], [0, 342 / 667, -3], [0, 0, 1]],
    )


def test_preprocessing_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plan_dust3r_preprocessing([])
    with pytest.raises(ValueError, match="224 or 512"):
        plan_dust3r_preprocessing([(10, 10)], image_size=518)
    with pytest.raises(ValueError, match="positive"):
        plan_dust3r_preprocessing([(0, 10)])
