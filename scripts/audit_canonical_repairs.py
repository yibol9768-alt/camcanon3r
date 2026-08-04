#!/usr/bin/env python3
"""Audit a multi-scene canonical-camera repair preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.repair_audit import audit_canonical_repairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("repaired_root", type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--source-variants", nargs="+", required=True)
    parser.add_argument("--prefix", default="canonical_")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_canonical_repairs(
        args.prepared_root,
        args.repaired_root,
        scenes=args.scenes,
        source_variants=args.source_variants,
        prefix=args.prefix,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
