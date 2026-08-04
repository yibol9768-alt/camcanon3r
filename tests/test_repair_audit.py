import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from camcanon3r.protocol import prepare_scene
from camcanon3r.repair import canonicalize_variant
from camcanon3r.repair_audit import audit_canonical_repairs


def _prepared_repair(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(2):
        pixels = np.zeros((12, 16, 3), dtype=np.uint8)
        pixels[..., index] = 100 + index
        Image.fromarray(pixels).save(source / f"view_{index}.png")
    prepared = tmp_path / "prepared/scene"
    prepare_scene(
        source,
        prepared,
        variants=["identity", "asymmetric_crop_075"],
        seed=17,
        scene_name="scene",
    )
    repaired = tmp_path / "repaired/scene"
    canonicalize_variant(prepared / "identity", repaired / "identity")
    canonicalize_variant(
        prepared / "asymmetric_crop_075",
        repaired / "canonical_asymmetric_crop_075",
    )
    return prepared.parent, repaired.parent


def test_audit_canonical_repairs_validates_pixels_masks_and_provenance(
    tmp_path: Path,
) -> None:
    prepared, repaired = _prepared_repair(tmp_path)
    report = audit_canonical_repairs(
        prepared,
        repaired,
        scenes=["scene"],
        source_variants=["identity", "asymmetric_crop_075"],
    )
    assert report["status"] == "complete"
    assert report["image_count"] == 4
    assert report["mask_count"] == 4
    assert report["identity_pixel_matches"] == 2
    assert report["fill_policy"] is None
    crop = report["valid_fraction_by_variant"]["canonical_asymmetric_crop_075"]
    assert 0.0 < crop["median"] < 1.0
    assert len(report["tree_sha256"]) == 64


def test_audit_canonical_repairs_rejects_mask_tampering(tmp_path: Path) -> None:
    prepared, repaired = _prepared_repair(tmp_path)
    mask = repaired / "scene/_masks/identity/view_0.png"
    Image.fromarray(np.zeros((12, 16), dtype=np.uint8)).save(mask)
    with pytest.raises(ValueError, match="valid fraction mismatch"):
        audit_canonical_repairs(
            prepared,
            repaired,
            scenes=["scene"],
            source_variants=["identity", "asymmetric_crop_075"],
        )


def test_audit_canonical_repairs_rejects_manifest_drift(tmp_path: Path) -> None:
    prepared, repaired = _prepared_repair(tmp_path)
    manifest_path = repaired / "scene/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"][1]["images"][0]["repair"]["input_variant"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="repair provenance mismatch"):
        audit_canonical_repairs(
            prepared,
            repaired,
            scenes=["scene"],
            source_variants=["identity", "asymmetric_crop_075"],
        )


def test_audit_canonical_repairs_enforces_registered_fill_policy(
    tmp_path: Path,
) -> None:
    prepared, repaired = _prepared_repair(tmp_path)
    report = audit_canonical_repairs(
        prepared,
        repaired,
        scenes=["scene"],
        source_variants=["identity", "asymmetric_crop_075"],
        fill_policy="neutral_gray",
    )
    assert report["fill_policy"] == "neutral_gray"

    with pytest.raises(ValueError, match="repair fill policy mismatch"):
        audit_canonical_repairs(
            prepared,
            repaired,
            scenes=["scene"],
            source_variants=["identity", "asymmetric_crop_075"],
            fill_policy="black",
        )
