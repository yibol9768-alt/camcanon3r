"""Seekable, block-cached HTTP range reads for large frozen ZIP archives."""

from __future__ import annotations

import io
import re
import time
from collections import OrderedDict
from typing import Protocol

import requests

_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class _RangeSession(Protocol):
    def get(self, url: str, **kwargs: object) -> requests.Response: ...

    def close(self) -> None: ...


class HTTPRangeReader(io.RawIOBase):
    """Expose one immutable HTTP object as a seekable binary file.

    The caller freezes the byte length and, when available, the ETag.  Every
    fetched block must return an exact ``206 Content-Range`` response matching
    that frozen identity.  This prevents a server that ignores Range or changes
    the object mid-run from turning a selective extraction into an unaudited
    full download or a mixed-version archive.
    """

    def __init__(
        self,
        url: str,
        *,
        size: int,
        etag: str | None = None,
        block_size: int = 4 * 1024 * 1024,
        cache_blocks: int = 8,
        timeout: tuple[float, float] = (30.0, 180.0),
        max_attempts: int = 12,
        retry_delay_seconds: float = 1.0,
        session: _RangeSession | None = None,
    ) -> None:
        super().__init__()
        if not url.startswith(("http://", "https://")):
            raise ValueError("HTTP range URL must use http:// or https://")
        if size <= 0:
            raise ValueError("HTTP object size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if cache_blocks <= 0:
            raise ValueError("cache_blocks must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        self.url = url
        self.size = int(size)
        self.etag = etag
        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)
        self.timeout = timeout
        self.max_attempts = int(max_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self._position = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._session = session or requests.Session()
        self._owns_session = session is None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self._position = min(int(target), self.size)
        return self._position

    def _fetch_block(self, block_index: int) -> bytes:
        cached = self._cache.pop(block_index, None)
        if cached is not None:
            self._cache[block_index] = cached
            return cached

        start = block_index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        if start >= self.size:
            return b""
        response = None
        last_error: Exception | None = None
        expected_length = end - start + 1
        for attempt in range(self.max_attempts):
            try:
                candidate = self._session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                last_error = error
            else:
                if candidate.status_code not in {408, 429} and not (
                    500 <= candidate.status_code < 600
                ):
                    if candidate.status_code == 206:
                        payload = bytes(candidate.content)
                        if len(payload) != expected_length:
                            last_error = OSError(
                                "short HTTP range response: "
                                f"expected={expected_length}, actual={len(payload)}"
                            )
                        else:
                            response = candidate
                            break
                    else:
                        response = candidate
                        break
                else:
                    last_error = requests.HTTPError(
                        f"transient HTTP status {candidate.status_code}"
                    )
            if attempt + 1 < self.max_attempts:
                delay = min(self.retry_delay_seconds * 2**attempt, 10.0)
                time.sleep(delay)
        if response is None:
            raise OSError(
                f"HTTP range failed after {self.max_attempts} attempts: {last_error}"
            ) from last_error
        if response.status_code != 206:
            raise OSError(
                "server did not honor the frozen byte range: "
                f"status={response.status_code}, range={start}-{end}"
            )
        content_range = response.headers.get("Content-Range", "")
        match = _CONTENT_RANGE.fullmatch(content_range)
        expected = (start, end, self.size)
        actual = tuple(int(value) for value in match.groups()) if match else None
        if actual != expected:
            raise OSError(
                f"unexpected Content-Range: expected={expected}, actual={actual}"
            )
        response_etag = response.headers.get("ETag")
        if self.etag is not None and response_etag != self.etag:
            raise OSError(
                f"HTTP ETag changed: expected={self.etag!r}, actual={response_etag!r}"
            )
        payload = bytes(response.content)
        if len(payload) != expected_length:
            raise OSError(
                "short HTTP range response: "
                f"expected={expected_length}, actual={len(payload)}"
            )
        self._cache[block_index] = payload
        while len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return payload

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        if self._position >= self.size or size == 0:
            return b""
        if size is None or size < 0:
            size = self.size - self._position
        remaining = min(int(size), self.size - self._position)
        chunks: list[bytes] = []
        while remaining:
            block_index = self._position // self.block_size
            block = self._fetch_block(block_index)
            offset = self._position - block_index * self.block_size
            take = min(remaining, len(block) - offset)
            if take <= 0:
                raise OSError("HTTP range cache returned no forward progress")
            chunks.append(block[offset : offset + take])
            self._position += take
            remaining -= take
        return b"".join(chunks)

    def close(self) -> None:
        if not self.closed:
            self._cache.clear()
            if self._owns_session:
                self._session.close()
        super().close()
