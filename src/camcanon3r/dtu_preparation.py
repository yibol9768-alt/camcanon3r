"""Prepare the frozen DTU sparse-view transform matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .prediction import write_json_atomic
from .protocol import list_images, prepare_scene


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dtu_selection(
    selection_root: Path,
    prepared_root: Path,
    protocol_path: Path,
    variant_config_path: Path,
    rectified_report_path: Path,
    *,
    resume: bool = False,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    variant_config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    extraction = json.loads(rectified_report_path.read_text(encoding="utf-8"))
    if not protocol.get("frozen_before_dtu_gt_inspection"):
        raise ValueError("DTU preparation protocol is not frozen")
    if extraction.get("status") != "complete":
        raise RuntimeError("DTU Rectified extraction report is not complete")
    scans = tuple(int(value) for value in protocol["evaluation_scans"])
    camera_ids = tuple(
        int(value) for value in protocol["rectified_archive_camera_ids_one_based"]
    )
    lighting = int(protocol["lighting_index"])
    variants = tuple(str(value) for value in variant_config["ordered_variants"])
    seed = int(variant_config["base_seed"])
    if int(variant_config["variant_seed_stride"]) != 10_007:
        raise ValueError("variant seed stride does not match prepare_scene")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("DTU variants must be non-empty and unique")

    expected_targets = {
        f"source/scan{scan}/rect_{camera_id:03d}_{lighting}_r5000.png"
        for scan in scans
        for camera_id in camera_ids
    }
    extraction_targets = {
        str(record["target"]) for record in extraction.get("members", [])
    }
    if extraction_targets != expected_targets:
        raise ValueError(
            "DTU Rectified extraction does not match the frozen design: "
            f"missing={sorted(expected_targets - extraction_targets)}, "
            f"extra={sorted(extraction_targets - expected_targets)}"
        )

    expected_scene_names = {f"scan{scan}" for scan in scans}
    source_root = selection_root / "source"
    actual_scene_names = {path.name for path in source_root.iterdir() if path.is_dir()}
    if actual_scene_names != expected_scene_names:
        raise ValueError(
            "DTU source scene design mismatch: "
            f"missing={sorted(expected_scene_names - actual_scene_names)}, "
            f"extra={sorted(actual_scene_names - expected_scene_names)}"
        )

    completed: list[str] = []
    for scan in scans:
        scene = f"scan{scan}"
        source_dir = source_root / scene
        expected_names = [
            f"rect_{camera_id:03d}_{lighting}_r5000.png" for camera_id in camera_ids
        ]
        actual_names = [path.name for path in list_images(source_dir)]
        if actual_names != expected_names:
            raise ValueError(
                f"DTU input views do not match the frozen design for {scene}: "
                f"expected={expected_names}, actual={actual_names}"
            )
        output_dir = prepared_root / scene
        if output_dir.exists() and any(output_dir.iterdir()) and not resume:
            raise FileExistsError(
                f"prepared DTU scene exists; use --resume: {output_dir}"
            )
        prepare_scene(
            source_dir,
            output_dir,
            variants=variants,
            seed=seed,
            scene_name=scene,
            max_views=len(camera_ids),
            resume=resume,
        )
        completed.append(scene)

    report = {
        "schema_version": "1.0",
        "status": "complete",
        "selection_root": str(selection_root.resolve()),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "variant_config": str(variant_config_path.resolve()),
        "variant_config_sha256": _sha256(variant_config_path),
        "rectified_extraction_report": str(rectified_report_path.resolve()),
        "rectified_extraction_report_sha256": _sha256(rectified_report_path),
        "scene_count": len(completed),
        "scenes": completed,
        "camera_ids_one_based": list(camera_ids),
        "lighting_index": lighting,
        "variant_count": len(variants),
        "variants": list(variants),
        "image_count": len(completed) * len(camera_ids) * len(variants),
    }
    write_json_atomic(prepared_root / "preparation_report.json", report)
    return report
