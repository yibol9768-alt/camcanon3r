#!/usr/bin/env python3
"""Audit the complete ETH3D mechanism preparation against its frozen prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.prepared_audit import audit_prepared_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_report", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--variant-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    if selection_report.get("completed_at") is None:
        raise RuntimeError("ETH3D selection report is incomplete")
    scene_images = {
        str(record["scene"]): [str(name) for name in record["image_names"]]
        for record in selection_report["selection"]["scenes"]
    }
    report = audit_prepared_sweep(
        args.prepared_root,
        args.variant_config,
        scene_images,
        reference_root=args.reference_root,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
