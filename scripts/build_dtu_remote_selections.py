#!/usr/bin/env python3
"""Build exact DTU extraction selections from complete remote ZIP indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.dtu_selection import build_dtu_remote_selections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--sources", type=Path, default=Path("configs/dtu_sources.json")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/dtu_protocol.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dtu_remote_selections(
        args.sources,
        args.protocol,
        args.index_root,
        args.output_root,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
