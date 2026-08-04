from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from camcanon3r.dtu_preparation import prepare_dtu_selection

SCANS = (
    1,
    4,
    9,
    10,
    11,
    12,
    13,
    15,
    23,
    24,
    29,
    32,
    33,
    34,
    48,
    49,
    62,
    75,
    77,
    110,
    114,
    118,
)
CAMERAS = (23, 26, 29)


def _design(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    selection = tmp_path / "selected"
    protocol = tmp_path / "protocol.json"
    variants = tmp_path / "variants.json"
    extraction = tmp_path / "rectified_report.json"
    prepared = tmp_path / "prepared"
    protocol.write_text(
        json.dumps(
            {
                "frozen_before_dtu_gt_inspection": True,
                "evaluation_scans": list(SCANS),
                "rectified_archive_camera_ids_one_based": list(CAMERAS),
                "lighting_index": 3,
            }
        ),
        encoding="utf-8",
    )
    variants.write_text(
        json.dumps(
            {
                "base_seed": 17,
                "variant_seed_stride": 10007,
                "ordered_variants": [
                    "identity",
                    "asymmetric_crop_075",
                    "shared_asymmetric_crop_075",
                ],
            }
        ),
        encoding="utf-8",
    )
    members = []
    for scan in SCANS:
        for camera in CAMERAS:
            target = f"source/scan{scan}/rect_{camera:03d}_3_r5000.png"
            path = selection / target
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 8), color=(scan % 255, camera, 3)).save(path)
            members.append({"target": target})
    extraction.write_text(
        json.dumps({"status": "complete", "members": members}),
        encoding="utf-8",
    )
    return selection, prepared, protocol, variants, extraction


def test_prepare_dtu_selection_enforces_and_renders_design(tmp_path: Path) -> None:
    selection, prepared, protocol, variants, extraction = _design(tmp_path)
    report = prepare_dtu_selection(
        selection,
        prepared,
        protocol,
        variants,
        extraction,
    )
    assert report["scene_count"] == 22
    assert report["variant_count"] == 3
    assert report["image_count"] == 198
    assert len(list(prepared.rglob("*.png"))) == 198
    manifest = json.loads((prepared / "scan1/manifest.json").read_text())
    shared = next(
        item
        for item in manifest["variants"]
        if item["name"] == "shared_asymmetric_crop_075"
    )
    assert shared["images"][0]["matrix"] == shared["images"][1]["matrix"]


def test_prepare_dtu_selection_rejects_incomplete_extraction(tmp_path: Path) -> None:
    selection, prepared, protocol, variants, extraction = _design(tmp_path)
    payload = json.loads(extraction.read_text())
    payload["members"].pop()
    extraction.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the frozen design"):
        prepare_dtu_selection(
            selection,
            prepared,
            protocol,
            variants,
            extraction,
        )
