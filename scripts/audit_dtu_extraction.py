#!/usr/bin/env python3
"""Independently rehash the complete frozen DTU selected tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.prediction import write_json_atomic
from camcanon3r.remote_zip_selection import audit_remote_zip_extractions

EXPECTED_MEMBER_COUNTS = {"sampleset": 58, "rectified": 66, "points": 22}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_remote_zip_extractions(
        {
            archive: (
                args.selection_root / f"{archive}.json",
                args.report_root / f"{archive}_extraction_report.json",
            )
            for archive in EXPECTED_MEMBER_COUNTS
        },
        args.output_root,
    )
    actual_counts = {
        archive: int(record["member_count"])
        for archive, record in report["archives"].items()
    }
    if actual_counts != EXPECTED_MEMBER_COUNTS or report["member_count"] != 146:
        raise ValueError(
            f"DTU extraction design drift: expected={EXPECTED_MEMBER_COUNTS}, "
            f"actual={actual_counts}"
        )
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "member_count": report["member_count"],
                "tree_sha256": report["tree_sha256"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
