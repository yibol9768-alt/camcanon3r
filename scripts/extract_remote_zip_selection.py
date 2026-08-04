#!/usr/bin/env python3
"""Extract a frozen subset of a remote ZIP with byte-range requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.remote_zip_selection import extract_remote_zip_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--block-size-mib", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size_mib <= 0:
        raise ValueError("--block-size-mib must be positive")
    report = extract_remote_zip_selection(
        args.selection,
        args.output_root,
        args.report,
        resume=args.resume,
        block_size=args.block_size_mib * 1024 * 1024,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "member_count": report["member_count"],
                "completed_members": report["completed_members"],
                "report": str(args.report),
            }
        )
    )


if __name__ == "__main__":
    main()
