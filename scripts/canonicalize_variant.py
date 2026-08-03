#!/usr/bin/env python3
"""Undo one known preprocessing variant onto a canonical camera canvas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.repair import FILL_POLICIES, canonicalize_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--fill-policy", choices=sorted(FILL_POLICIES), default="neutral_gray"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = canonicalize_variant(
        args.variant_dir, args.output_dir, fill_policy=args.fill_policy
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
