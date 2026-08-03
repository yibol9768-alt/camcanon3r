import numpy as np

from camcanon3r.vggt_preprocess import plan_vggt_preprocessing


def test_crop_mode_records_resize_and_portrait_center_crop() -> None:
    landscape, portrait = plan_vggt_preprocessing(
        [(800, 400), (400, 800)], mode="crop"
    )
    assert landscape.resized_size == (518, 252)
    assert landscape.padding == (0, 133, 0, 133)
    assert landscape.affine.target_size == (518, 518)
    assert portrait.resized_size == (518, 1036)
    assert portrait.crop_top == 259
    np.testing.assert_allclose(
        portrait.affine.matrix,
        [[518 / 400, 0, 0], [0, 1036 / 800, -259], [0, 0, 1]],
    )


def test_pad_mode_records_official_multiple_of_14_and_padding() -> None:
    (spec,) = plan_vggt_preprocessing([(800, 400)], mode="pad")
    assert spec.resized_size == (518, 252)
    assert spec.padding == (0, 133, 0, 133)
    assert spec.affine.target_size == (518, 518)
    np.testing.assert_allclose(
        spec.affine.matrix,
        [[518 / 800, 0, 0], [0, 252 / 400, 133], [0, 0, 1]],
    )
