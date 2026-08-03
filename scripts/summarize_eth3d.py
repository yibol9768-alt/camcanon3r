#!/usr/bin/env python3
"""Aggregate nested ETH3D scene evaluations with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.summary import summarize_eth3d_evaluations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.results_root.rglob("*_vs_gt.json"))
    summary = summarize_eth3d_evaluations(
        paths,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    rendered = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
