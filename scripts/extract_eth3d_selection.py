#!/usr/bin/env python3
"""Extract a deterministic four-view subset from verified ETH3D archives."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from camcanon3r.downloads import load_download_manifest, sha256_file
from camcanon3r.eth3d_archives import (
    build_eth3d_selection,
    parse_7z_slt,
    selected_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--views-per-scene", type=int, default=4)
    parser.add_argument(
        "--seven-zip",
        type=Path,
        default=Path("/mnt/c/Program Files/7-Zip/7z.exe"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _windows_path(path: Path) -> str:
    result = subprocess.run(
        ["wslpath", "-w", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _list_members(seven_zip: Path, archive: Path) -> dict[str, int]:
    result = subprocess.run(
        [str(seven_zip), "l", "-slt", _windows_path(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_7z_slt(result.stdout)


def _extract_members(
    seven_zip: Path,
    archive: Path,
    output_root: Path,
    members: list[str],
    expected_sizes: dict[str, int],
) -> None:
    pending = [
        member
        for member in members
        if not _valid_size(output_root / member, expected_sizes[member])
    ]
    if not pending:
        return
    list_root = output_root / ".selection_lists"
    list_root.mkdir(parents=True, exist_ok=True)
    list_path = list_root / f"{archive.name}.txt"
    list_path.write_text(
        "\n".join(member.replace("/", "\\") for member in pending) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(seven_zip),
            "x",
            "-y",
            "-aoa",
            "-scsUTF-8",
            f"-o{_windows_path(output_root)}",
            _windows_path(archive),
            f"@{_windows_path(list_path)}",
        ],
        check=True,
    )


def _valid_size(path: Path, expected_bytes: int | None) -> bool:
    if not path.is_file():
        return False
    return expected_bytes is None or path.stat().st_size == expected_bytes


def _records_by_path(selection: dict[str, object]) -> dict[str, int]:
    records: dict[str, int] = {}
    fields = (
        "raw",
        "undistorted",
        "depth",
        "raw_calibration",
        "undistorted_calibration",
    )
    for scene in selection["scenes"]:
        for field in fields:
            for record in scene[field]:
                path = str(record["path"])
                expected_bytes = int(record["bytes"])
                if path in records and records[path] != expected_bytes:
                    raise ValueError(f"conflicting selected size for {path}")
                records[path] = expected_bytes
    return records


def _verified_downloads(
    manifest_path: Path, archive_root: Path
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    manifest, items = load_download_manifest(manifest_path)
    report_path = archive_root / "download_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("completed_at") is None:
        raise RuntimeError("ETH3D download report is not complete")
    reports = {
        str(record["filename"]): record for record in report["archives"]
    }
    expected_names = {item.filename for item in items}
    if set(reports) != expected_names:
        raise RuntimeError("download report does not cover the frozen manifest")
    for item in items:
        record = reports[item.filename]
        archive = archive_root / item.filename
        if record.get("status") != "verified":
            raise RuntimeError(f"archive is not verified: {item.filename}")
        if not _valid_size(archive, item.expected_bytes):
            raise RuntimeError(f"verified archive is missing or wrong-sized: {archive}")
    return manifest, reports


def main() -> None:
    args = parse_args()
    if not args.seven_zip.is_file():
        raise FileNotFoundError(f"7-Zip executable is missing: {args.seven_zip}")
    manifest, download_reports = _verified_downloads(
        args.manifest, args.archive_root
    )
    scenes = list(manifest["scenes"])
    undistorted_archive = (
        args.archive_root / "multi_view_training_dslr_undistorted.7z"
    )
    raw_archive = args.archive_root / "multi_view_training_dslr_jpg.7z"
    undistorted_members = _list_members(args.seven_zip, undistorted_archive)
    raw_members = _list_members(args.seven_zip, raw_archive)
    depth_archives = {
        scene: args.archive_root / f"{scene}_dslr_depth.7z" for scene in scenes
    }
    depth_members = {
        scene: _list_members(args.seven_zip, archive)
        for scene, archive in depth_archives.items()
    }
    selection = build_eth3d_selection(
        scenes=scenes,
        undistorted_members=undistorted_members,
        raw_members=raw_members,
        depth_members=depth_members,
        views_per_scene=args.views_per_scene,
    )

    report_path = args.output_root / "selection_report.json"
    existing_report = None
    if report_path.exists():
        if not args.resume:
            raise FileExistsError(f"selection report already exists: {report_path}")
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        if existing_report.get("selection") != selection:
            raise ValueError("existing ETH3D selection report does not match")
    elif args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output root is not empty; use --resume: {args.output_root}"
        )

    expected = _records_by_path(selection)
    remaining_bytes = sum(
        size
        for path, size in expected.items()
        if not _valid_size(args.output_root / path, size)
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_root).free
    reserve_bytes = 5 * 1024**3
    if free_bytes < remaining_bytes + reserve_bytes:
        raise RuntimeError(
            "insufficient free space for selected extraction plus 5 GiB reserve"
        )

    raw_selection = selected_paths(selection, "raw") + selected_paths(
        selection, "raw_calibration"
    )
    undistorted_selection = selected_paths(
        selection, "undistorted"
    ) + selected_paths(selection, "undistorted_calibration")
    _extract_members(
        args.seven_zip,
        undistorted_archive,
        args.output_root,
        undistorted_selection,
        expected,
    )
    _extract_members(
        args.seven_zip, raw_archive, args.output_root, raw_selection, expected
    )
    for scene_record in selection["scenes"]:
        scene = str(scene_record["scene"])
        _extract_members(
            args.seven_zip,
            depth_archives[scene],
            args.output_root,
            [str(record["path"]) for record in scene_record["depth"]],
            expected,
        )

    invalid = [
        path
        for path, expected_bytes in expected.items()
        if not _valid_size(args.output_root / path, expected_bytes)
    ]
    if invalid:
        raise RuntimeError(f"selected extraction is incomplete: {invalid}")
    extracted_records = [
        {
            "path": path,
            "bytes": expected_bytes,
            "sha256": sha256_file(args.output_root / path),
        }
        for path, expected_bytes in sorted(expected.items())
    ]
    report = {
        "schema_version": "1.0",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "archive_root": str(args.archive_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "download_archives": [
            download_reports[name] for name in sorted(download_reports)
        ],
        "selection": selection,
        "extracted_files": extracted_records,
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "status": "complete",
                "scene_count": len(scenes),
                "file_count": len(extracted_records),
                "report": str(report_path),
                "resumed": existing_report is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
