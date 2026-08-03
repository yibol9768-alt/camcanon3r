#!/usr/bin/env python3
"""Summarize all comparison JSON files beneath a results directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.summary import summarize_comparison_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rotation-threshold", type=float, default=2.0)
    parser.add_argument("--depth-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.results_root.rglob("*_vs_identity.json"))
    summary = summarize_comparison_files(
        paths,
        rotation_threshold=args.rotation_threshold,
        depth_threshold=args.depth_threshold,
    )
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
