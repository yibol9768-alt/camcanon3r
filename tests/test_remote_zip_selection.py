from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import requests

from camcanon3r.remote_zip_selection import (
    extract_remote_zip_selection,
    load_remote_zip_selection,
)


class _Response:
    def __init__(self, content: bytes, start: int, end: int, total: int) -> None:
        self.status_code = 206
        self.content = content
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{total}",
            "ETag": '"frozen"',
        }


class _Session:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def get(self, url: str, **kwargs: object) -> requests.Response:
        assert url == "https://example.test/archive.zip"
        range_header = kwargs["headers"]["Range"]  # type: ignore[index]
        start_text, end_text = str(range_header).removeprefix("bytes=").split("-")
        start, end = int(start_text), int(end_text)
        return _Response(
            self.payload[start : end + 1], start, end, len(self.payload)
        )  # type: ignore[return-value]

    def close(self) -> None:
        pass


def _archive() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source/one.txt", b"first payload")
        archive.writestr("source/two.txt", b"second payload")
    return stream.getvalue()


def _selection(path: Path, payload: bytes, *, target: str = "scan/one.txt") -> None:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo("source/one.txt")
    path.write_text(
        json.dumps(
            {
                "url": "https://example.test/archive.zip",
                "expected_bytes": len(payload),
                "etag": "frozen",
                "members": [
                    {
                        "source": info.filename,
                        "target": target,
                        "bytes": info.file_size,
                        "crc32": f"{info.CRC:08x}",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_extract_remote_zip_selection_is_atomic_and_resumable(
    tmp_path: Path,
) -> None:
    payload = _archive()
    selection = tmp_path / "selection.json"
    output = tmp_path / "output"
    report = tmp_path / "report.json"
    _selection(selection, payload)

    first = extract_remote_zip_selection(
        selection,
        output,
        report,
        block_size=64,
        session=_Session(payload),
    )
    assert first["status"] == "complete"
    assert (output / "scan/one.txt").read_bytes() == b"first payload"
    assert not (output / "scan/.one.txt.part").exists()

    second = extract_remote_zip_selection(
        selection,
        output,
        report,
        resume=True,
        block_size=64,
        session=_Session(payload),
    )
    assert second == first

    (output / "scan/one.txt").unlink()
    third = extract_remote_zip_selection(
        selection,
        output,
        report,
        resume=True,
        block_size=64,
        session=_Session(payload),
    )
    assert len(third["members"]) == 1
    assert (output / "scan/one.txt").read_bytes() == b"first payload"


def test_remote_zip_selection_rejects_unsafe_target(tmp_path: Path) -> None:
    payload = _archive()
    selection = tmp_path / "selection.json"
    _selection(selection, payload, target="../escape.txt")
    with pytest.raises(ValueError, match="safe relative path"):
        load_remote_zip_selection(selection)
