import json
import os
import subprocess
import sys
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


def test_download_launcher_exposes_src_package(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    archive_root = tmp_path / "archives"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "configs").mkdir()
    (repo_root / ".venv" / "bin").mkdir(parents=True)
    (repo_root / "src" / "camcanon3r").mkdir(parents=True)

    (repo_root / "src" / "camcanon3r" / "__init__.py").write_text(
        "MARKER = 'imported-from-src'\n", encoding="utf-8"
    )
    (repo_root / "configs" / "eth3d_training_archives.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (repo_root / "scripts" / "download_archives.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "import camcanon3r\n"
        "Path(sys.argv[-1]).write_text(camcanon3r.MARKER, encoding='utf-8')\n",
        encoding="utf-8",
    )
    proxy_wrapper = repo_root / "scripts" / "with_download_proxy.sh"
    proxy_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    proxy_wrapper.chmod(0o755)
    (repo_root / ".venv" / "bin" / "python").symlink_to(sys.executable)

    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "start_eth3d_download_my5090.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "CAMCANON3R_REPO_ROOT": str(repo_root),
            "CAMCANON3R_ETH3D_ARCHIVES": str(archive_root),
            "PYTHONPATH": "",
        }
    )
    subprocess.run([str(launcher)], check=True, env=env)

    assert (archive_root / "download_report.json").read_text(
        encoding="utf-8"
    ) == "imported-from-src"
