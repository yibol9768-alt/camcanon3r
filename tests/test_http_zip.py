from __future__ import annotations

import io
import zipfile

import pytest
import requests

from camcanon3r.http_zip import HTTPRangeReader


class _Response:
    def __init__(
        self,
        *,
        status_code: int,
        content: bytes,
        headers: dict[str, str],
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers


class _Session:
    def __init__(
        self,
        payload: bytes,
        *,
        etag: str = '"frozen"',
        failures: int = 0,
    ) -> None:
        self.payload = payload
        self.etag = etag
        self.failures = failures
        self.requests: list[tuple[int, int]] = []
        self.closed = False

    def get(self, url: str, **kwargs: object) -> _Response:
        assert url == "https://example.test/archive.zip"
        if self.failures:
            self.failures -= 1
            raise requests.exceptions.ProxyError("transient proxy failure")
        range_header = kwargs["headers"]["Range"]  # type: ignore[index]
        start_text, end_text = str(range_header).removeprefix("bytes=").split("-")
        start, end = int(start_text), int(end_text)
        self.requests.append((start, end))
        return _Response(
            status_code=206,
            content=self.payload[start : end + 1],
            headers={
                "Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
                "ETag": self.etag,
            },
        )

    def close(self) -> None:
        self.closed = True


def _zip_payload() -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("scan1/image.png", b"image-one" * 100)
        archive.writestr("scan4/image.png", b"image-four" * 100)
    return target.getvalue()


def test_http_range_reader_supports_zipfile_and_reuses_blocks() -> None:
    payload = _zip_payload()
    session = _Session(payload)
    with HTTPRangeReader(
        "https://example.test/archive.zip",
        size=len(payload),
        etag='"frozen"',
        block_size=128,
        cache_blocks=8,
        session=session,
    ) as source:
        with zipfile.ZipFile(source) as archive:
            assert archive.namelist() == ["scan1/image.png", "scan4/image.png"]
            assert archive.read("scan4/image.png") == b"image-four" * 100
        request_count = len(session.requests)
        source.seek(0)
        assert source.read(10) == payload[:10]
        assert len(session.requests) == request_count
    assert not session.closed


def test_http_range_reader_rejects_identity_drift() -> None:
    payload = _zip_payload()
    session = _Session(payload, etag='"changed"')
    source = HTTPRangeReader(
        "https://example.test/archive.zip",
        size=len(payload),
        etag='"frozen"',
        block_size=128,
        session=session,
    )
    with pytest.raises(OSError, match="ETag changed"):
        source.read(1)


def test_http_range_reader_retries_transient_proxy_failure() -> None:
    payload = _zip_payload()
    session = _Session(payload, failures=2)
    source = HTTPRangeReader(
        "https://example.test/archive.zip",
        size=len(payload),
        etag='"frozen"',
        block_size=128,
        max_attempts=3,
        retry_delay_seconds=0,
        session=session,
    )
    assert source.read(1) == payload[:1]
