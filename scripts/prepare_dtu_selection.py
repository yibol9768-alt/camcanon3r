#!/usr/bin/env python3
"""Prepare every frozen DTU scan under the registered transform matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.dtu_preparation import prepare_dtu_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_root", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("rectified_extraction_report", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/dtu_protocol.json")
    )
    parser.add_argument(
        "--variant-config",
        type=Path,
        default=Path("configs/eth3d_mechanism_variants.json"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_dtu_selection(
        args.selection_root,
        args.prepared_root,
        args.protocol,
        args.variant_config,
        args.rectified_extraction_report,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scene_count": report["scene_count"],
                "variant_count": report["variant_count"],
                "image_count": report["image_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
