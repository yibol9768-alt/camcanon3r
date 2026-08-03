import json
from pathlib import Path

import pytest

from camcanon3r.downloads import (
    DownloadItem,
    inspect_download,
    load_download_manifest,
    sha256_file,
)


def test_download_manifest_and_resume_states(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archives": [
                    {
                        "filename": "sample.7z",
                        "url": "https://example.test/sample.7z",
                        "expected_bytes": 4,
                        "purpose": "test",
                    }
                ]
            }
        )
    )
    _, items = load_download_manifest(manifest)
    item = items[0]
    missing = inspect_download(item, tmp_path)
    assert missing["status"] == "missing"
    assert missing["remaining_bytes"] == 4

    (tmp_path / "sample.7z.part").write_bytes(b"ab")
    resumable = inspect_download(item, tmp_path)
    assert resumable["status"] == "resumable"
    assert resumable["downloaded_bytes"] == 2

    (tmp_path / "sample.7z.part").unlink()
    final = tmp_path / "sample.7z"
    final.write_bytes(b"abcd")
    complete = inspect_download(item, tmp_path)
    assert complete["status"] == "complete"
    assert sha256_file(final) == (
        "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"
    )


def test_download_validation_rejects_unsafe_or_conflicting_records(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archives": [
                    {
                        "filename": "../escape.7z",
                        "url": "https://example.test/escape.7z",
                        "expected_bytes": 1,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="basename"):
        load_download_manifest(manifest)

    item = DownloadItem("bad.7z", "https://example.test/bad.7z", 3, "")
    (tmp_path / "bad.7z").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="wrong size"):
        inspect_download(item, tmp_path)


def test_frozen_eth3d_manifest_covers_all_training_scenes() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "eth3d_training_archives.json"
    )
    payload, items = load_download_manifest(manifest_path)
    assert len(payload["scenes"]) == 13
    assert len(set(payload["scenes"])) == 13
    assert len(items) == 15
    assert sum(item.expected_bytes for item in items) == 16_873_121_856
    assert "no scene was selected using model outcomes" in payload[
        "selection_policy"
    ]
