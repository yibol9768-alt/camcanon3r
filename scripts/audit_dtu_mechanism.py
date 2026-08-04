#!/usr/bin/env python3
"""Audit the frozen prepared DTU mechanism sweep bytewise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.dtu_audit import audit_dtu_preparation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dtu_preparation(
        args.prepared_root,
        args.protocol,
        args.variant_config,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
