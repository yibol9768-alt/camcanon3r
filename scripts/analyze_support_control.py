#!/usr/bin/env python3
"""Analyze the frozen cross-dataset support-preserving control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.prediction import write_json_atomic
from camcanon3r.support_analysis import analyze_support_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--summary",
        action="append",
        nargs=3,
        metavar=("MODEL", "DATASET", "PATH"),
        required=True,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_support_control(
        [(model, dataset, Path(path)) for model, dataset, path in args.summary],
        args.protocol,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "promotion_gate": report["promotion_gate"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
