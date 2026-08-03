#!/usr/bin/env python3
"""Compute auditable GT gap recovery for one matched repair result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.repair_evaluation import evaluate_repair_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("identity", type=Path)
    parser.add_argument("corrupt", type=Path)
    parser.add_argument("repaired", type=Path)
    parser.add_argument("--clean-control", type=Path)
    parser.add_argument("--minimum-gap", type=float, default=1e-12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    result = evaluate_repair_records(
        _read(args.identity),
        _read(args.corrupt),
        _read(args.repaired),
        clean_control=(
            None if args.clean_control is None else _read(args.clean_control)
        ),
        minimum_gap=args.minimum_gap,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
