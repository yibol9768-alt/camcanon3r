#!/usr/bin/env python3
"""Compare two VGGT camera predictions using gauge-invariant pairwise poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.comparison import compare_vggt_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-label", default="identity")
    parser.add_argument("--candidate-label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_vggt_predictions(
        args.reference,
        args.candidate,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
