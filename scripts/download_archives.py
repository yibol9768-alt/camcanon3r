#!/usr/bin/env python3
"""Download and verify every archive in a frozen manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from camcanon3r.downloads import (
    DownloadItem,
    inspect_download,
    load_download_manifest,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--only", nargs="+")
    return parser.parse_args()


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _download(item: DownloadItem, partial_path: Path) -> None:
    subprocess.run(
        [
            "curl",
            "-fL",
            "--retry",
            "30",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--speed-time",
            "90",
            "--speed-limit",
            "1024",
            "-C",
            "-",
            "-o",
            str(partial_path),
            item.url,
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    manifest, items = load_download_manifest(args.manifest)
    if args.only:
        requested = set(args.only)
        known = {item.filename for item in items}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown --only archive names: {unknown}")
        items = [item for item in items if item.filename in requested]

    args.destination.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.destination / "download_report.json"
    inspections = {
        item.filename: inspect_download(item, args.destination) for item in items
    }
    remaining_bytes = sum(
        int(inspection["remaining_bytes"]) for inspection in inspections.values()
    )
    free_bytes = shutil.disk_usage(args.destination).free
    reserve_bytes = 5 * 1024**3
    if free_bytes < remaining_bytes + reserve_bytes:
        raise RuntimeError(
            "insufficient free space for downloads plus 5 GiB reserve: "
            f"need {remaining_bytes + reserve_bytes}, have {free_bytes}"
        )

    report: dict[str, object] = {
        "schema_version": "1.0",
        "manifest": str(args.manifest.resolve()),
        "manifest_metadata": {
            key: value for key, value in manifest.items() if key != "archives"
        },
        "destination": str(args.destination.resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "archives": [],
    }
    archive_reports: list[dict[str, object]] = []
    report["archives"] = archive_reports
    for item in items:
        inspection = inspect_download(item, args.destination)
        final_path = Path(inspection["path"])
        partial_path = Path(inspection["partial_path"])
        print(
            json.dumps(
                {
                    "archive": item.filename,
                    "status": inspection["status"],
                    "downloaded_bytes": inspection["downloaded_bytes"],
                    "expected_bytes": item.expected_bytes,
                }
            ),
            flush=True,
        )
        if inspection["status"] != "complete":
            _download(item, partial_path)
            actual_bytes = partial_path.stat().st_size
            if actual_bytes != item.expected_bytes:
                raise RuntimeError(
                    f"downloaded size mismatch for {item.filename}: "
                    f"expected {item.expected_bytes}, got {actual_bytes}"
                )
            partial_path.replace(final_path)
        digest = sha256_file(final_path)
        archive_reports.append(
            {
                "filename": item.filename,
                "url": item.url,
                "purpose": item.purpose,
                "expected_bytes": item.expected_bytes,
                "actual_bytes": final_path.stat().st_size,
                "sha256": digest,
                "status": "verified",
            }
        )
        _write_report(report_path, report)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_report(report_path, report)
    print(json.dumps({"status": "complete", "report": str(report_path)}))


if __name__ == "__main__":
    main()
