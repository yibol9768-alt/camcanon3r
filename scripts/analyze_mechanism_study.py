#!/usr/bin/env python3
"""Analyze frozen severity and crop-scope mechanism summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.mechanism_analysis import analyze_mechanism_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--summary",
        nargs=3,
        action="append",
        metavar=("MODEL", "DATASET", "PATH"),
        required=True,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--rotation-threshold", type=float, default=2.0)
    parser.add_argument("--depth-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_mechanism_summaries(
        [(model, dataset, Path(path)) for model, dataset, path in args.summary],
        args.variant_config,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
        rotation_threshold=args.rotation_threshold,
        depth_threshold=args.depth_threshold,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "analysis_count": len(report["analyses"]),
                "family_support": report["family_support"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
