import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from camcanon3r.orbit_preparation import (
    audit_canonical_orbit_scene,
    load_orbit_protocol,
    prepare_canonical_orbit_scene,
)

PROTOCOL = Path("configs/orbit_projection_protocol.json")


def _write_canonical_scene(root: Path) -> None:
    variant = "canonical_asymmetric_crop_075"
    records = []
    for index, name in enumerate(("one.png", "two.png")):
        height, width = 5 + index, 7 + index
        yy, xx = np.mgrid[:height, :width]
        pixels = np.stack(
            [
                (xx + 17 * index) % 256,
                (yy * 7 + 29 * index) % 256,
                (xx * 11 + yy * 13) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[1:-1, 1:-1] = 255
        image_path = root / variant / name
        mask_path = root / "_masks" / variant / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, mode="RGB").save(image_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        records.append(
            {
                "source": name,
                "output": f"{variant}/{name}",
                "source_size": [width, height],
                "target_size": [width, height],
                "matrix": np.eye(3).tolist(),
                "repair": {
                    "method": "known_affine_inverse_canvas",
                    "input_variant": "asymmetric_crop_075",
                    "fill_policy": "neutral_gray",
                    "valid_mask": f"_masks/{variant}/{name}",
                },
            }
        )
    manifest = {
        "protocol_version": "0.2.0",
        "scene": "synthetic",
        "variants": [
            {
                "name": variant,
                "interpolation": "test fixture",
                "images": records,
            }
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def test_load_orbit_protocol_validates_frozen_design():
    protocol = load_orbit_protocol(PROTOCOL)

    assert protocol["method_name"] == "canonical_orbit_projection"
    assert len(protocol["orbit"]["ordered_members"]) == 9


def test_prepare_and_audit_preserve_every_decoded_byte(tmp_path):
    source = tmp_path / "canonical"
    output = tmp_path / "orbit"
    _write_canonical_scene(source)

    manifest = prepare_canonical_orbit_scene(source, output, PROTOCOL)
    report = audit_canonical_orbit_scene(source, output, PROTOCOL)

    assert [record["name"] for record in manifest["variants"]] == [
        "orbit_center",
        "orbit_left",
        "orbit_right",
        "orbit_top",
        "orbit_bottom",
        "orbit_top_left",
        "orbit_bottom_right",
        "orbit_top_right",
        "orbit_bottom_left",
    ]
    assert report["status"] == "complete"
    assert report["member_count"] == 9
    assert report["image_count"] == report["mask_count"] == 18
    assert report["decoded_rgb_matches"] == 18
    assert report["decoded_mask_matches"] == 18

    center = manifest["variants"][0]["images"][0]
    left = manifest["variants"][1]["images"][0]
    right = manifest["variants"][2]["images"][0]
    assert center["target_size"] == [9, 7]
    assert center["orbit"]["offset_xy"] == [1, 1]
    assert left["orbit"]["offset_xy"] == [0, 1]
    assert right["orbit"]["offset_xy"] == [2, 1]


def test_resume_is_idempotent_and_reaudited(tmp_path):
    source = tmp_path / "canonical"
    output = tmp_path / "orbit"
    _write_canonical_scene(source)
    prepare_canonical_orbit_scene(source, output, PROTOCOL)
    before = audit_canonical_orbit_scene(source, output, PROTOCOL)

    prepare_canonical_orbit_scene(source, output, PROTOCOL, resume=True)
    after = audit_canonical_orbit_scene(source, output, PROTOCOL)

    assert before["tree_sha256"] == after["tree_sha256"]


def test_audit_rejects_a_changed_visible_pixel(tmp_path):
    source = tmp_path / "canonical"
    output = tmp_path / "orbit"
    _write_canonical_scene(source)
    prepare_canonical_orbit_scene(source, output, PROTOCOL)
    path = output / "orbit_center" / "one.png"
    with Image.open(path) as opened:
        pixels = np.asarray(opened.convert("RGB")).copy()
    pixels[1, 1, 0] ^= 1
    Image.fromarray(pixels, mode="RGB").save(path)

    with pytest.raises(ValueError, match="decoded source RGB byte"):
        audit_canonical_orbit_scene(source, output, PROTOCOL)


def test_protocol_rejects_nonpreserving_design(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["orbit"]["resample_visible_pixels"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen support-preserving"):
        load_orbit_protocol(path)
