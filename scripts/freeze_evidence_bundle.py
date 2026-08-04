#!/usr/bin/env python3
"""Freeze a compact evidence bundle while hashing but not copying large outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

METADATA_FILES = {Path("BUNDLE.json"), Path("SHA256SUMS")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"bundle target must be a safe relative path: {path}")
    return path


def _source_path(value: object, manifest_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, manifest_path.resolve().parent.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _expand_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "1.0"
        or not manifest.get("frozen_before_final_dtu_results")
    ):
        raise ValueError("evidence bundle manifest is not a frozen schema-1.0 design")
    entries: list[dict[str, Any]] = []
    for record in manifest.get("files", []):
        source = _source_path(record["source"], manifest_path)
        if not source.is_file():
            raise FileNotFoundError(f"bundle source file is missing: {source}")
        entries.append(
            {
                "mode": "copy",
                "source": source.resolve(),
                "target": _safe_relative(record["target"]),
            }
        )
    for tree in manifest.get("trees", []):
        source_root = _source_path(tree["source"], manifest_path)
        if not source_root.is_dir():
            raise FileNotFoundError(f"bundle source tree is missing: {source_root}")
        mode = str(tree["mode"])
        if mode not in {"copy", "hash_only"}:
            raise ValueError(f"unsupported bundle tree mode: {mode}")
        paths = sorted(
            path for path in source_root.glob(str(tree["glob"])) if path.is_file()
        )
        expected_count = int(tree["expected_count"])
        if len(paths) != expected_count:
            raise ValueError(
                f"bundle tree count mismatch for {source_root}: "
                f"expected={expected_count}, actual={len(paths)}"
            )
        target_root = _safe_relative(tree["target"])
        for path in paths:
            entries.append(
                {
                    "mode": mode,
                    "source": path.resolve(),
                    "target": _safe_relative(
                        target_root / path.relative_to(source_root)
                    ),
                }
            )
    targets = [entry["target"] for entry in entries]
    if len(set(targets)) != len(targets) or set(targets) & METADATA_FILES:
        raise ValueError("evidence bundle contains duplicate or reserved targets")
    if not entries:
        raise ValueError("evidence bundle manifest contains no files")
    return manifest, entries


def _copy_or_validate(source: Path, target: Path, *, resume: bool) -> str:
    source_sha256 = _sha256(source)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        if not resume or not temporary.is_file() or _sha256(temporary) != source_sha256:
            raise ValueError(f"partial bundle copy cannot be resumed: {temporary}")
        if target.exists():
            if not target.is_file() or _sha256(target) != source_sha256:
                raise ValueError(f"resumed bundle target differs from source: {target}")
            temporary.unlink()
        else:
            temporary.replace(target)
    if target.exists():
        if not resume:
            raise FileExistsError(f"bundle target exists; use --resume: {target}")
        if not target.is_file() or _sha256(target) != source_sha256:
            raise ValueError(f"resumed bundle target differs from source: {target}")
        return source_sha256
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temporary)
    if _sha256(temporary) != source_sha256:
        raise RuntimeError(f"bundle copy SHA-256 mismatch: {source}")
    temporary.replace(target)
    return source_sha256


def _write_or_validate(path: Path, content: bytes, *, resume: bool) -> None:
    expected = hashlib.sha256(content).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        if not resume or not temporary.is_file() or _sha256(temporary) != expected:
            raise ValueError(f"partial bundle metadata cannot be resumed: {temporary}")
        if path.exists():
            if _sha256(path) != expected:
                raise ValueError(f"existing bundle metadata differs: {path}")
            temporary.unlink()
        else:
            temporary.replace(path)
    if path.exists():
        if not resume or _sha256(path) != expected:
            raise ValueError(f"existing bundle metadata differs: {path}")
        return
    temporary.write_bytes(content)
    temporary.replace(path)


def freeze_bundle(
    manifest_path: Path, output_root: Path, *, resume: bool = False
) -> dict[str, object]:
    manifest, entries = _expand_manifest(manifest_path)
    if output_root.exists() and not resume:
        raise FileExistsError(f"bundle output exists; use --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    expected_copies = {entry["target"] for entry in entries if entry["mode"] == "copy"}
    actual = {
        path.relative_to(output_root)
        for path in output_root.rglob("*")
        if path.is_file()
    }
    temporary_files = {
        path
        for path in actual
        if path.name.startswith(".") and path.name.endswith(".tmp")
    }
    unexpected = actual - expected_copies - METADATA_FILES - temporary_files
    if unexpected:
        raise ValueError(
            f"bundle output contains unexpected files: {sorted(map(str, unexpected))}"
        )

    records: list[dict[str, object]] = []
    copied_count = 0
    hashed_only_count = 0
    for entry in entries:
        source = Path(entry["source"])
        target = Path(entry["target"])
        if entry["mode"] == "copy":
            sha256 = _copy_or_validate(source, output_root / target, resume=resume)
            copied_count += 1
        else:
            sha256 = _sha256(source)
            hashed_only_count += 1
        records.append(
            {
                "mode": entry["mode"],
                "source": str(source),
                "target": target.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256,
            }
        )

    bundle = {
        "schema_version": "1.0",
        "status": "complete",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "frozen_before_final_dtu_results": manifest["frozen_before_final_dtu_results"],
        "entry_count": len(records),
        "copied_count": copied_count,
        "hashed_only_count": hashed_only_count,
        "records": records,
    }
    bundle_bytes = json.dumps(bundle, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _write_or_validate(output_root / "BUNDLE.json", bundle_bytes, resume=resume)
    checksum_paths = sorted(expected_copies | {Path("BUNDLE.json")})
    checksum_lines = [
        f"{_sha256(output_root / path)}  {path.as_posix()}" for path in checksum_paths
    ]
    checksum_bytes = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    _write_or_validate(output_root / "SHA256SUMS", checksum_bytes, resume=resume)

    final_actual = {
        path.relative_to(output_root)
        for path in output_root.rglob("*")
        if path.is_file()
    }
    expected_final = expected_copies | METADATA_FILES
    if final_actual != expected_final:
        raise RuntimeError(
            "final bundle file design mismatch: "
            f"missing={sorted(map(str, expected_final - final_actual))}, "
            f"extra={sorted(map(str, final_actual - expected_final))}"
        )
    return bundle


def main() -> None:
    args = parse_args()
    report = freeze_bundle(args.manifest, args.output_root, resume=args.resume)
    print(
        json.dumps(
            {
                "status": report["status"],
                "entry_count": report["entry_count"],
                "copied_count": report["copied_count"],
                "hashed_only_count": report["hashed_only_count"],
                "output": str(args.output_root),
            }
        )
    )


if __name__ == "__main__":
    main()
