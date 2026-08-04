"""Strict audit of a prepared DTU transform sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .prediction import write_json_atomic
from .prepared_audit import audit_prepared_sweep


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_dtu_preparation(
    prepared_root: Path,
    protocol_path: Path,
    variant_config_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    preparation_path = prepared_root / "preparation_report.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    scans = [int(value) for value in protocol["evaluation_scans"]]
    camera_ids = [
        int(value) for value in protocol["rectified_archive_camera_ids_one_based"]
    ]
    lighting = int(protocol["lighting_index"])
    variants = [str(value) for value in config["ordered_variants"]]
    expected_report = {
        "status": "complete",
        "protocol_sha256": _sha256(protocol_path),
        "variant_config_sha256": _sha256(variant_config_path),
        "scene_count": len(scans),
        "scenes": [f"scan{scan}" for scan in scans],
        "camera_ids_one_based": camera_ids,
        "lighting_index": lighting,
        "variant_count": len(variants),
        "variants": variants,
        "image_count": len(scans) * len(camera_ids) * len(variants),
    }
    actual_report = {key: preparation.get(key) for key in expected_report}
    if actual_report != expected_report:
        raise ValueError(
            "DTU preparation report does not match the frozen design: "
            f"expected={expected_report}, actual={actual_report}"
        )
    image_names = [
        f"rect_{camera_id:03d}_{lighting}_r5000.png" for camera_id in camera_ids
    ]
    scene_images = {f"scan{scan}": image_names for scan in scans}
    report = audit_prepared_sweep(
        prepared_root,
        variant_config_path,
        scene_images,
    )
    report.update(
        {
            "dataset": "DTU MVS Data Set 2014",
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": expected_report["protocol_sha256"],
            "variant_config": str(variant_config_path.resolve()),
            "variant_config_sha256": expected_report["variant_config_sha256"],
            "preparation_report": str(preparation_path.resolve()),
            "preparation_report_sha256": _sha256(preparation_path),
            "camera_ids_one_based": camera_ids,
            "lighting_index": lighting,
        }
    )
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
