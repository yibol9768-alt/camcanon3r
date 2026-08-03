from pathlib import Path

import numpy as np

from camcanon3r.comparison import compare_vggt_predictions


def test_identical_vggt_predictions_have_zero_disagreement(tmp_path: Path) -> None:
    extrinsic = np.stack(
        [
            np.column_stack([np.eye(3), np.zeros(3)]),
            np.column_stack([np.eye(3), np.array([1.0, 0.0, 0.0])]),
        ]
    )
    depth = np.arange(1, 9, dtype=np.float64).reshape(2, 2, 2)
    affine = np.repeat(np.eye(3)[None], 2, axis=0)
    reference = tmp_path / "identity.npz"
    candidate = tmp_path / "repeat.npz"
    for path in (reference, candidate):
        np.savez(
            path,
            extrinsic=extrinsic,
            depth=depth,
            source_to_model_affine=affine,
        )

    result = compare_vggt_predictions(reference, candidate)
    assert result["rotation_degrees"]["median"] == 0.0
    assert result["translation_direction_degrees"]["median"] == 0.0
    assert result["aligned_depth_consistency"]["mean_abs_rel"] == 0.0
