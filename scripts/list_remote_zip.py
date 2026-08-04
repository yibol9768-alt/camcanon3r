#!/usr/bin/env python3
"""Inspect a frozen HTTP ZIP through byte ranges without downloading it."""

from __future__ import annotations

import argparse
import fnmatch
import json
import zipfile

from camcanon3r.http_zip import HTTPRangeReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--etag")
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--block-size-mib", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    with HTTPRangeReader(
        args.url,
        size=args.expected_bytes,
        etag=args.etag,
        block_size=args.block_size_mib * 1024 * 1024,
    ) as source, zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        matches = [
            info for info in infos if fnmatch.fnmatch(info.filename, args.pattern)
        ]
        payload = {
            "status": "complete",
            "url": args.url,
            "expected_bytes": args.expected_bytes,
            "etag": args.etag,
            "member_count": len(infos),
            "compressed_payload_bytes": sum(info.compress_size for info in infos),
            "uncompressed_payload_bytes": sum(info.file_size for info in infos),
            "pattern": args.pattern,
            "match_count": len(matches),
            "matches": [
                {
                    "path": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                }
                for info in matches[: args.limit]
            ],
            "matches_truncated": len(matches) > args.limit,
        }
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
