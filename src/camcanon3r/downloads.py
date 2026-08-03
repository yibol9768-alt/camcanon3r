"""Validated, resumable downloads for machine-local research artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DownloadItem:
    filename: str
    url: str
    expected_bytes: int
    purpose: str


def load_download_manifest(path: Path) -> tuple[dict[str, Any], list[DownloadItem]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    archives = payload.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError("download manifest must contain a non-empty archives list")
    items: list[DownloadItem] = []
    filenames: set[str] = set()
    for record in archives:
        filename = str(record["filename"])
        if Path(filename).name != filename:
            raise ValueError(f"archive filename must be a basename: {filename!r}")
        if filename in filenames:
            raise ValueError(f"duplicate archive filename: {filename}")
        filenames.add(filename)
        url = str(record["url"])
        if not url.startswith("https://"):
            raise ValueError(f"archive URL must use HTTPS: {url}")
        expected_bytes = int(record["expected_bytes"])
        if expected_bytes <= 0:
            raise ValueError("expected_bytes must be positive")
        items.append(
            DownloadItem(
                filename=filename,
                url=url,
                expected_bytes=expected_bytes,
                purpose=str(record.get("purpose", "")),
            )
        )
    return payload, items


def inspect_download(item: DownloadItem, destination: Path) -> dict[str, object]:
    final_path = destination / item.filename
    partial_path = destination / f"{item.filename}.part"
    if final_path.exists():
        actual_bytes = final_path.stat().st_size
        if actual_bytes != item.expected_bytes:
            raise ValueError(
                f"completed file has wrong size for {item.filename}: "
                f"expected {item.expected_bytes}, got {actual_bytes}"
            )
        return {
            "status": "complete",
            "path": final_path,
            "partial_path": partial_path,
            "downloaded_bytes": actual_bytes,
            "remaining_bytes": 0,
        }
    partial_bytes = partial_path.stat().st_size if partial_path.exists() else 0
    if partial_bytes > item.expected_bytes:
        raise ValueError(
            f"partial file exceeds expected size for {item.filename}: "
            f"expected at most {item.expected_bytes}, got {partial_bytes}"
        )
    return {
        "status": "resumable" if partial_bytes else "missing",
        "path": final_path,
        "partial_path": partial_path,
        "downloaded_bytes": partial_bytes,
        "remaining_bytes": item.expected_bytes - partial_bytes,
    }


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
