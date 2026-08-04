import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from camcanon3r.dtu_audit import audit_dtu_preparation
from camcanon3r.protocol import prepare_scene


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    protocol_path = tmp_path / "protocol.json"
    config_path = tmp_path / "variants.json"
    protocol = {
        "evaluation_scans": [1],
        "rectified_archive_camera_ids_one_based": [23, 26, 29],
        "lighting_index": 3,
    }
    config = {
        "ordered_variants": ["identity", "center_crop_075"],
        "variant_seeds": {"identity": 17, "center_crop_075": 10024},
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    names = [f"rect_{camera_id:03d}_3_r5000.png" for camera_id in (23, 26, 29)]
    for index, name in enumerate(names):
        Image.fromarray(np.full((8, 10, 3), 20 + index, dtype=np.uint8)).save(
            source / name
        )
    prepared_root = tmp_path / "prepared"
    prepare_scene(
        source,
        prepared_root / "scan1",
        variants=config["ordered_variants"],
        seed=17,
        scene_name="scan1",
    )
    (prepared_root / "preparation_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol_sha256": _sha256(protocol_path),
                "variant_config_sha256": _sha256(config_path),
                "scene_count": 1,
                "scenes": ["scan1"],
                "camera_ids_one_based": [23, 26, 29],
                "lighting_index": 3,
                "variant_count": 2,
                "variants": config["ordered_variants"],
                "image_count": 6,
            }
        ),
        encoding="utf-8",
    )
    return prepared_root, protocol_path, config_path


def test_audit_dtu_preparation_binds_report_and_image_tree(tmp_path: Path) -> None:
    prepared, protocol, config = _fixture(tmp_path)
    report = audit_dtu_preparation(prepared, protocol, config)
    assert report["status"] == "complete"
    assert report["scene_count"] == 1
    assert report["png_count"] == 6
    assert report["camera_ids_one_based"] == [23, 26, 29]
    assert len(report["tree_sha256"]) == 64


def test_audit_dtu_preparation_rejects_report_drift(tmp_path: Path) -> None:
    prepared, protocol, config = _fixture(tmp_path)
    report_path = prepared / "preparation_report.json"
    report = json.loads(report_path.read_text())
    report["image_count"] = 5
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the frozen design"):
        audit_dtu_preparation(prepared, protocol, config)
