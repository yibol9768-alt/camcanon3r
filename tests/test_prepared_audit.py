from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from camcanon3r.prepared_audit import audit_prepared_sweep
from camcanon3r.protocol import prepare_scene


def _source(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (12, 8), color=(index * 20, 30, 40)).save(
            path / f"image_{index}.jpg"
        )
    return path


def test_audit_prepared_sweep_matches_prefix_and_shared_window(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    reference = tmp_path / "reference"
    sources = tmp_path / "sources"
    scene_images = {}
    full_variants = (
        "identity",
        "asymmetric_crop_075",
        "shared_asymmetric_crop_075",
    )
    for scene in ("first", "second"):
        source = _source(sources, scene)
        scene_images[scene] = [f"image_{index}.jpg" for index in range(3)]
        prepare_scene(
            source,
            prepared / scene,
            variants=full_variants,
            seed=17,
            scene_name=scene,
        )
        prepare_scene(
            source,
            reference / scene,
            variants=full_variants[:2],
            seed=17,
            scene_name=scene,
        )
    config = tmp_path / "variants.json"
    config.write_text(
        json.dumps(
            {
                "ordered_variants": list(full_variants),
                "variant_seeds": {
                    "identity": 17,
                    "asymmetric_crop_075": 10024,
                    "shared_asymmetric_crop_075": 20031,
                },
            }
        ),
        encoding="utf-8",
    )
    report = audit_prepared_sweep(
        prepared,
        config,
        scene_images,
        reference_root=reference,
        output_path=tmp_path / "audit.json",
    )
    assert report["png_count"] == 18
    assert report["reference_image_matches"] == 12
    assert report["shared_variant_scene_count"] == 2


def test_audit_prepared_sweep_rejects_extra_file(tmp_path: Path) -> None:
    source = _source(tmp_path / "sources", "first")
    prepared = tmp_path / "prepared"
    prepare_scene(
        source,
        prepared / "first",
        variants=("identity",),
        seed=17,
        scene_name="first",
    )
    config = tmp_path / "variants.json"
    config.write_text(
        json.dumps(
            {
                "ordered_variants": ["identity"],
                "variant_seeds": {"identity": 17},
            }
        ),
        encoding="utf-8",
    )
    (prepared / "first/extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="prepared file design mismatch"):
        audit_prepared_sweep(
            prepared,
            config,
            {"first": [f"image_{index}.jpg" for index in range(3)]},
        )
