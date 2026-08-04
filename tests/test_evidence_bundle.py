import json
from pathlib import Path

import pytest

from scripts.freeze_evidence_bundle import freeze_bundle


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "single.json").write_text('{"value": 1}\n', encoding="utf-8")
    copied = source / "copied"
    copied.mkdir()
    for index in range(2):
        (copied / f"record_{index}.json").write_text(
            json.dumps({"index": index}) + "\n", encoding="utf-8"
        )
    large = source / "large"
    large.mkdir()
    (large / "prediction.npz").write_bytes(b"large prediction placeholder")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frozen_before_final_dtu_results": True,
                "files": [
                    {"source": str(source / "single.json"), "target": "single.json"}
                ],
                "trees": [
                    {
                        "source": str(copied),
                        "target": "records",
                        "glob": "*.json",
                        "expected_count": 2,
                        "mode": "copy",
                    },
                    {
                        "source": str(large),
                        "target": "prediction_hashes",
                        "glob": "*.npz",
                        "expected_count": 1,
                        "mode": "hash_only",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, source, tmp_path / "bundle"


def test_evidence_bundle_copies_small_files_and_hashes_large_files(
    tmp_path: Path,
) -> None:
    manifest, _, output = _fixture(tmp_path)
    report = freeze_bundle(manifest, output)
    assert report["copied_count"] == 3
    assert report["hashed_only_count"] == 1
    assert (output / "single.json").is_file()
    assert (output / "records/record_0.json").is_file()
    assert not (output / "prediction_hashes/prediction.npz").exists()
    assert (output / "BUNDLE.json").is_file()
    assert (output / "SHA256SUMS").is_file()

    resumed = freeze_bundle(manifest, output, resume=True)
    assert resumed == report


def test_evidence_bundle_refuses_changed_or_extra_resumed_files(
    tmp_path: Path,
) -> None:
    manifest, source, output = _fixture(tmp_path)
    freeze_bundle(manifest, output)
    (source / "single.json").write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from source"):
        freeze_bundle(manifest, output, resume=True)

    (source / "single.json").write_text('{"value": 1}\n', encoding="utf-8")
    (output / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        freeze_bundle(manifest, output, resume=True)
