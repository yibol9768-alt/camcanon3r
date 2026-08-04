#!/usr/bin/env python3
"""Audit a prepared ETH3D or DTU support-preserving control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.support_control import audit_support_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--variant-config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--eth3d-selection-report", type=Path)
    source.add_argument("--dtu-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _scene_images(args: argparse.Namespace) -> dict[str, list[str]]:
    if args.eth3d_selection_report is not None:
        report = json.loads(args.eth3d_selection_report.read_text(encoding="utf-8"))
        if report.get("completed_at") is None:
            raise RuntimeError("ETH3D selection report is incomplete")
        return {
            str(record["scene"]): [str(name) for name in record["image_names"]]
            for record in report["selection"]["scenes"]
        }
    protocol = json.loads(args.dtu_protocol.read_text(encoding="utf-8"))
    camera_ids = [
        int(value) for value in protocol["rectified_archive_camera_ids_one_based"]
    ]
    lighting = int(protocol["lighting_index"])
    names = [f"rect_{camera_id:03d}_{lighting}_r5000.png" for camera_id in camera_ids]
    return {f"scan{int(scan)}": names for scan in protocol["evaluation_scans"]}


def main() -> None:
    args = parse_args()
    report = audit_support_control(
        args.prepared_root,
        args.reference_root,
        args.variant_config,
        _scene_images(args),
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
