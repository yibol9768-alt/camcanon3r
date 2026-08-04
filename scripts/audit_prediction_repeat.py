#!/usr/bin/env python3
"""Audit an exact-input prediction repeat across a frozen scene set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.prediction_repeat import audit_prediction_repeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--variant", default="identity")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_prediction_repeat(
        args.reference_root,
        args.candidate_root,
        scenes=args.scenes,
        variant=args.variant,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scene_count": report["scene_count"],
                "exact_scene_count": report["exact_scene_count"],
                "array_count": report["array_count"],
                "exact_array_count": report["exact_array_count"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
