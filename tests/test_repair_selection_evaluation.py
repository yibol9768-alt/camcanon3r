import json
from pathlib import Path

import pytest

from scripts.evaluate_repair_selection import _load_scene_records


def _write(path: Path, *, scene: str, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scene": scene,
                "variant": variant,
                "relative_rotation_degrees": {"median": 1.0},
            }
        ),
        encoding="utf-8",
    )


def test_load_repair_scene_records_requires_paired_roots(tmp_path: Path) -> None:
    original = tmp_path / "original"
    repaired = tmp_path / "repaired"
    for scene in ("first", "second"):
        _write(
            original / scene / "identity_vs_gt.json",
            scene=scene,
            variant="identity",
        )
        _write(
            original / scene / "asymmetric_crop_075_vs_gt.json",
            scene=scene,
            variant="asymmetric_crop_075",
        )
        _write(
            repaired / scene / "identity_vs_gt.json",
            scene=scene,
            variant="identity",
        )
        _write(
            repaired / scene / "canonical_asymmetric_crop_075_vs_gt.json",
            scene=scene,
            variant="canonical_asymmetric_crop_075",
        )
    records = _load_scene_records(
        original,
        repaired,
        identity_variant="identity",
        corrupt_variant="asymmetric_crop_075",
        clean_control_variant="identity",
        repaired_variant="canonical_asymmetric_crop_075",
    )
    assert sorted(records) == ["first", "second"]
    assert records["first"][1]["variant"] == "asymmetric_crop_075"

    (repaired / "second/canonical_asymmetric_crop_075_vs_gt.json").unlink()
    with pytest.raises(FileNotFoundError, match="repair evaluation is missing"):
        _load_scene_records(
            original,
            repaired,
            identity_variant="identity",
            corrupt_variant="asymmetric_crop_075",
            clean_control_variant="identity",
            repaired_variant="canonical_asymmetric_crop_075",
        )
