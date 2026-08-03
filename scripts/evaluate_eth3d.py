#!/usr/bin/env python3
"""Evaluate one VGGT result against official ETH3D poses and depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.eth3d import evaluate_eth3d_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("calibration_dir", type=Path)
    parser.add_argument("depth_dir", nargs="?", type=Path)
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="evaluate pose only, for pre-undistorted images without aligned depth GT",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.skip_depth and args.depth_dir is not None:
        parser.error("depth_dir cannot be supplied together with --skip-depth")
    if not args.skip_depth and args.depth_dir is None:
        parser.error("depth_dir is required unless --skip-depth is used")
    return args


def main() -> None:
    args = parse_args()
    result = evaluate_eth3d_prediction(
        args.prediction,
        args.calibration_dir,
        depth_dir=None if args.skip_depth else args.depth_dir,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
