"""Auditable selective extraction from immutable remote ZIP archives."""

from __future__ import annotations

import hashlib
import json
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .http_zip import HTTPRangeReader, _RangeSession
from .prediction import write_json_atomic


@dataclass(frozen=True)
class RemoteZipMember:
    source: str
    target: PurePosixPath
    expected_bytes: int
    crc32: int


@dataclass(frozen=True)
class RemoteZipSelection:
    url: str
    expected_bytes: int
    etag: str | None
    members: tuple[RemoteZipMember, ...]


def _relative_target(value: object) -> PurePosixPath:
    text = str(value)
    target = PurePosixPath(text)
    if (
        target.is_absolute()
        or not target.parts
        or any(part in {"", ".", ".."} for part in target.parts)
        or "\\" in text
    ):
        raise ValueError(f"member target must be a safe relative path: {value!r}")
    return target


def _normalized_etag(value: object) -> str | None:
    if value is None:
        return None
    etag = str(value)
    if not etag.startswith(('"', 'W/"')):
        etag = f'"{etag}"'
    return etag


def load_remote_zip_selection(path: Path) -> RemoteZipSelection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("remote ZIP selection must be a JSON object")
    url = str(payload["url"])
    if not url.startswith("https://"):
        raise ValueError("remote ZIP URL must use HTTPS")
    expected_bytes = int(payload["expected_bytes"])
    if expected_bytes <= 0:
        raise ValueError("remote ZIP expected_bytes must be positive")
    raw_members = payload.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("remote ZIP selection requires a non-empty members list")

    members: list[RemoteZipMember] = []
    sources: set[str] = set()
    targets: set[PurePosixPath] = set()
    for record in raw_members:
        if not isinstance(record, Mapping):
            raise TypeError("every remote ZIP member must be a JSON object")
        source = str(record["source"])
        target = _relative_target(record["target"])
        member_bytes = int(record["bytes"])
        if member_bytes < 0:
            raise ValueError("member bytes must be non-negative")
        crc_text = str(record["crc32"])
        if len(crc_text) != 8:
            raise ValueError(f"member CRC-32 must use eight hex digits: {crc_text!r}")
        try:
            crc32 = int(crc_text, 16)
        except ValueError as error:
            raise ValueError(f"invalid member CRC-32: {crc_text!r}") from error
        if source in sources:
            raise ValueError(f"duplicate remote ZIP source: {source}")
        if target in targets:
            raise ValueError(f"duplicate remote ZIP target: {target}")
        sources.add(source)
        targets.add(target)
        members.append(
            RemoteZipMember(
                source=source,
                target=target,
                expected_bytes=member_bytes,
                crc32=crc32,
            )
        )
    return RemoteZipSelection(
        url=url,
        expected_bytes=expected_bytes,
        etag=_normalized_etag(payload.get("etag")),
        members=tuple(members),
    )


def _hash_and_crc(path: Path) -> tuple[int, str, int]:
    digest = hashlib.sha256()
    crc = 0
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)
    return size, digest.hexdigest(), crc & 0xFFFFFFFF


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def extract_remote_zip_selection(
    selection_path: Path,
    output_root: Path,
    report_path: Path,
    *,
    resume: bool = False,
    block_size: int = 4 * 1024 * 1024,
    session: _RangeSession | None = None,
) -> dict[str, object]:
    """Extract exact frozen members and checkpoint hashes after every member."""

    selection = load_remote_zip_selection(selection_path)
    selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "in_progress",
        "selection": str(selection_path.resolve()),
        "selection_sha256": selection_sha256,
        "archive": {
            "url": selection.url,
            "expected_bytes": selection.expected_bytes,
            "etag": selection.etag,
        },
        "member_count": len(selection.members),
        "members": [],
    }
    if report_path.exists():
        if not resume:
            raise FileExistsError(f"extraction report already exists: {report_path}")
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("selection_sha256") != selection_sha256:
            raise ValueError("existing extraction report uses a different selection")
        report = previous
    report_members = report.get("members")
    if not isinstance(report_members, list):
        raise TypeError("extraction report members must be a list")
    expected_targets = {member.target.as_posix() for member in selection.members}
    completed: dict[str, Any] = {}
    for record in report_members:
        target_key = str(record["target"])
        if target_key not in expected_targets:
            raise ValueError(
                f"extraction report contains an extra target: {target_key}"
            )
        if target_key in completed:
            raise ValueError(
                f"extraction report contains a duplicate target: {target_key}"
            )
        completed[target_key] = record

    output_root.mkdir(parents=True, exist_ok=True)
    with (
        HTTPRangeReader(
            selection.url,
            size=selection.expected_bytes,
            etag=selection.etag,
            block_size=block_size,
            session=session,
        ) as source,
        zipfile.ZipFile(source) as archive,
    ):
        archive_infos = {info.filename: info for info in archive.infolist()}
        for member in selection.members:
            info = archive_infos.get(member.source)
            if info is None:
                raise FileNotFoundError(
                    f"remote ZIP member is missing: {member.source}"
                )
            if info.is_dir():
                raise ValueError(
                    f"selected remote ZIP member is a directory: {member.source}"
                )
            if info.file_size != member.expected_bytes or info.CRC != member.crc32:
                raise ValueError(
                    f"remote ZIP member identity mismatch: {member.source}"
                )

        for member in selection.members:
            target_key = member.target.as_posix()
            target = output_root.joinpath(*member.target.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            previous = completed.get(target_key)
            if target.exists():
                if not resume:
                    raise FileExistsError(f"selected output already exists: {target}")
                size, sha256, crc32 = _hash_and_crc(target)
                if size != member.expected_bytes or crc32 != member.crc32:
                    raise ValueError(f"existing selected output is invalid: {target}")
                if previous is not None and previous.get("sha256") != sha256:
                    raise ValueError(
                        f"existing output changed after extraction: {target}"
                    )
                if previous is None:
                    record = {
                        "source": member.source,
                        "target": target_key,
                        "bytes": size,
                        "crc32": f"{crc32:08x}",
                        "sha256": sha256,
                    }
                    report["members"].append(record)
                    completed[target_key] = record
                    _write_report(report_path, report)
                continue

            temporary = target.with_name(f".{target.name}.part")
            if temporary.exists():
                if not resume:
                    raise FileExistsError(
                        f"partial selected output already exists: {temporary}"
                    )
                temporary.unlink()
            digest = hashlib.sha256()
            crc = 0
            size = 0
            with archive.open(member.source) as remote, temporary.open("xb") as local:
                while chunk := remote.read(8 * 1024 * 1024):
                    local.write(chunk)
                    size += len(chunk)
                    crc = zlib.crc32(chunk, crc)
                    digest.update(chunk)
                local.flush()
            crc &= 0xFFFFFFFF
            if size != member.expected_bytes or crc != member.crc32:
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"extracted member failed identity check: {member.source}"
                )
            temporary.replace(target)
            record = {
                "source": member.source,
                "target": target_key,
                "bytes": size,
                "crc32": f"{crc:08x}",
                "sha256": digest.hexdigest(),
            }
            if previous is not None:
                report["members"] = [
                    item
                    for item in report["members"]
                    if str(item["target"]) != target_key
                ]
            report["members"].append(record)
            completed[target_key] = record
            _write_report(report_path, report)

    if set(completed) != expected_targets:
        raise RuntimeError("remote ZIP extraction did not complete the frozen design")
    report["status"] = "complete"
    report["completed_members"] = len(completed)
    _write_report(report_path, report)
    return report


def audit_remote_zip_extractions(
    archives: Mapping[str, tuple[Path, Path]],
    output_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Rehash completed selective extractions and reject tree drift."""

    if not archives:
        raise ValueError("at least one remote ZIP extraction is required")
    expected_targets: set[str] = set()
    archive_reports: dict[str, object] = {}
    tree_records: list[tuple[str, int, str, str]] = []
    for archive_id, (selection_path, report_path) in archives.items():
        selection = load_remote_zip_selection(selection_path)
        selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_archive = {
            "url": selection.url,
            "expected_bytes": selection.expected_bytes,
            "etag": selection.etag,
        }
        if (
            report.get("schema_version") != "1.0"
            or report.get("status") != "complete"
            or Path(str(report.get("selection"))).resolve() != selection_path.resolve()
            or report.get("selection_sha256") != selection_sha256
            or report.get("archive") != expected_archive
            or report.get("member_count") != len(selection.members)
            or report.get("completed_members") != len(selection.members)
        ):
            raise ValueError(
                f"incomplete or mismatched extraction report: {report_path}"
            )
        reported_members = report.get("members")
        if not isinstance(reported_members, list) or len(reported_members) != len(
            selection.members
        ):
            raise ValueError(f"extraction report member count drift: {report_path}")
        if any(not isinstance(record, Mapping) for record in reported_members):
            raise TypeError(f"extraction report member is not an object: {report_path}")
        expected_order = [member.target.as_posix() for member in selection.members]
        actual_order = [str(record.get("target")) for record in reported_members]
        if actual_order != expected_order:
            raise ValueError(f"extraction report member order drift: {report_path}")

        archive_digest = hashlib.sha256()
        for member, record in zip(selection.members, reported_members, strict=True):
            target_key = member.target.as_posix()
            if target_key in expected_targets:
                raise ValueError(
                    f"duplicate extraction target across archives: {target_key}"
                )
            expected_targets.add(target_key)
            target = output_root.joinpath(*member.target.parts)
            size, sha256, crc32 = _hash_and_crc(target)
            expected_record = {
                "source": member.source,
                "target": target_key,
                "bytes": member.expected_bytes,
                "crc32": f"{member.crc32:08x}",
                "sha256": sha256,
            }
            if (
                record != expected_record
                or size != member.expected_bytes
                or crc32 != member.crc32
            ):
                raise ValueError(f"extracted member identity drift: {target}")
            encoded = target_key.encode("utf-8") + b"\0"
            archive_digest.update(encoded)
            archive_digest.update(bytes.fromhex(sha256))
            tree_records.append((target_key, size, f"{crc32:08x}", sha256))
        archive_reports[archive_id] = {
            "selection": str(selection_path.resolve()),
            "selection_sha256": selection_sha256,
            "report": str(report_path.resolve()),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "member_count": len(selection.members),
            "tree_sha256": archive_digest.hexdigest(),
        }

    actual_targets = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if actual_targets != expected_targets:
        raise ValueError(
            "selected extraction tree drift: "
            f"missing={sorted(expected_targets - actual_targets)}, "
            f"extra={sorted(actual_targets - expected_targets)}"
        )
    combined_digest = hashlib.sha256()
    for target, size, crc32, sha256 in sorted(tree_records):
        combined_digest.update(target.encode("utf-8") + b"\0")
        combined_digest.update(str(size).encode("ascii") + b"\0")
        combined_digest.update(crc32.encode("ascii") + b"\0")
        combined_digest.update(bytes.fromhex(sha256))
    result = {
        "schema_version": "remote-zip-extraction-audit-1.0",
        "status": "complete",
        "output_root": str(output_root.resolve()),
        "archive_count": len(archives),
        "member_count": len(expected_targets),
        "tree_sha256": combined_digest.hexdigest(),
        "archives": archive_reports,
    }
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result
