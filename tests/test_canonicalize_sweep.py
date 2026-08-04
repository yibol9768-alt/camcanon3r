import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from camcanon3r.protocol import prepare_scene
from scripts.canonicalize_sweep import main


def _prepared(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(2):
        Image.fromarray(np.full((12, 16, 3), index * 50, dtype=np.uint8)).save(
            source / f"view_{index}.png"
        )
    prepared = tmp_path / "prepared/scene"
    prepare_scene(
        source,
        prepared,
        variants=["identity", "asymmetric_crop_075"],
        seed=17,
        scene_name="scene",
    )
    return prepared.parent


def _arguments(prepared: Path, repaired: Path, report: Path) -> list[str]:
    return [
        "canonicalize_sweep.py",
        str(prepared),
        str(repaired),
        "--scenes",
        "scene",
        "--variants",
        "identity",
        "asymmetric_crop_075",
        "--fill-policy",
        "neutral_gray",
        "--report",
        str(report),
        "--resume",
    ]


def test_canonicalize_sweep_checkpoints_complete_resumable_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    repaired = tmp_path / "repaired"
    report_path = tmp_path / "compute.json"
    monkeypatch.setattr(sys, "argv", _arguments(prepared, repaired, report_path))
    main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["record_count"] == 2
    assert report["canonicalization_seconds"]["count"] == 2
    assert report["canonicalization_seconds"]["total"] > 0.0

    monkeypatch.setattr(sys, "argv", _arguments(prepared, repaired, report_path))
    main()
    resumed = json.loads(report_path.read_text(encoding="utf-8"))
    assert resumed == report


def test_canonicalize_sweep_refuses_unaccounted_resumed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path)
    repaired = tmp_path / "repaired"
    report_path = tmp_path / "compute.json"
    monkeypatch.setattr(sys, "argv", _arguments(prepared, repaired, report_path))
    main()
    report_path.unlink()
    monkeypatch.setattr(sys, "argv", _arguments(prepared, repaired, report_path))
    with pytest.raises(RuntimeError, match="no resumable timing record"):
        main()
