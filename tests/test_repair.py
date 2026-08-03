import json
from pathlib import Path

import numpy as np
from PIL import Image

from camcanon3r.protocol import prepare_scene, protocol_affines
from camcanon3r.repair import canonicalize_variant


def _write_scene(path: Path) -> None:
    path.mkdir()
    for index in range(2):
        pixels = np.zeros((8, 10, 3), dtype=np.uint8)
        pixels[..., 0] = 30 + 10 * index
        pixels[2:6, 3:7, 1] = 220
        Image.fromarray(pixels).save(path / f"view_{index}.png")


def test_identity_canonicalization_is_exact_and_records_identity(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_scene(scene)
    prepared = tmp_path / "prepared"
    prepare_scene(scene, prepared, variants=["identity"], seed=17)
    repaired_root = tmp_path / "repaired"
    repaired_dir = repaired_root / "canonical_identity"
    result = canonicalize_variant(
        prepared / "identity", repaired_dir, fill_policy="neutral_gray"
    )

    source_pixels = np.asarray(Image.open(scene / "view_0.png"))
    repaired_pixels = np.asarray(Image.open(repaired_dir / "view_0.png"))
    np.testing.assert_array_equal(repaired_pixels, source_pixels)
    assert result["images"][0]["repair"]["valid_fraction"] == 1.0
    matrices = protocol_affines(
        repaired_dir, sorted(repaired_dir.glob("*.png"))
    )
    np.testing.assert_allclose(matrices, np.repeat(np.eye(3)[None], 2, axis=0))


def test_crop_canonicalization_restores_canvas_and_exports_mask(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_scene(scene)
    prepared = tmp_path / "prepared"
    prepare_scene(scene, prepared, variants=["center_crop_050"], seed=17)
    repaired_root = tmp_path / "repaired"
    repaired_dir = repaired_root / "canonical_center_crop_050"
    result = canonicalize_variant(
        prepared / "center_crop_050",
        repaired_dir,
        fill_policy="neutral_gray",
    )

    repaired = np.asarray(Image.open(repaired_dir / "view_0.png"))
    mask = np.asarray(
        Image.open(
            repaired_root / "_masks/canonical_center_crop_050/view_0.png"
        )
    )
    assert repaired.shape == (8, 10, 3)
    np.testing.assert_array_equal(repaired[0, 0], [127, 127, 127])
    assert mask[0, 0] == 0
    assert mask[4, 5] == 255
    valid_fraction = result["images"][0]["repair"]["valid_fraction"]
    assert 0.0 < valid_fraction < 0.5
    manifest = json.loads((repaired_root / "manifest.json").read_text())
    assert manifest["variants"][0]["name"] == "canonical_center_crop_050"
    assert manifest["variants"][0]["images"][0]["repair"]["input_variant"] == (
        "center_crop_050"
    )
