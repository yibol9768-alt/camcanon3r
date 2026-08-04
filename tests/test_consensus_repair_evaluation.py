import argparse
import json
from pathlib import Path

import pytest

from scripts.evaluate_consensus_repair import _load_scenes


def _evaluation(path: Path, *, scene: str, variant: str) -> None:
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


def _fixture(tmp_path: Path):
    original = tmp_path / "original"
    clean = tmp_path / "clean"
    protocol = {
        "source_variant": "asymmetric_crop_075",
        "candidate_order": [
            {"label": "gray", "repaired_variant": "canonical_gray"},
            {"label": "black", "repaired_variant": "canonical_black"},
            {"label": "mean", "repaired_variant": "canonical_mean"},
        ],
    }
    scene = "office"
    _evaluation(
        original / scene / "identity_vs_gt.json",
        scene=scene,
        variant="identity",
    )
    _evaluation(
        original / scene / "asymmetric_crop_075_vs_gt.json",
        scene=scene,
        variant="asymmetric_crop_075",
    )
    _evaluation(
        clean / scene / "identity_vs_gt.json",
        scene=scene,
        variant="identity",
    )
    candidates = []
    for record in protocol["candidate_order"]:
        predictions = tmp_path / f"predictions-{record['label']}"
        results = tmp_path / f"results-{record['label']}"
        prediction = predictions / scene / f"{record['repaired_variant']}.npz"
        prediction.parent.mkdir(parents=True)
        prediction.touch()
        prediction.with_suffix(".json").write_text("{}", encoding="utf-8")
        _evaluation(
            results / scene / f"{record['repaired_variant']}_vs_gt.json",
            scene=scene,
            variant=record["repaired_variant"],
        )
        candidates.append(
            [
                record["label"],
                record["repaired_variant"],
                str(predictions),
                str(results),
            ]
        )
    args = argparse.Namespace(
        original_results=original,
        clean_results=clean,
        candidate=candidates,
        identity_variant="identity",
    )
    return args, protocol


def test_load_consensus_scenes_enforces_frozen_candidates(tmp_path: Path) -> None:
    args, protocol = _fixture(tmp_path)
    scenes, candidates = _load_scenes(args, protocol)
    assert list(scenes) == ["office"]
    assert [record["label"] for record in candidates] == ["gray", "black", "mean"]
    assert set(scenes["office"]["candidates"]) == {"gray", "black", "mean"}

    args.candidate = list(reversed(args.candidate))
    with pytest.raises(ValueError, match="frozen protocol order"):
        _load_scenes(args, protocol)


def test_load_consensus_scenes_rejects_candidate_scene_drift(tmp_path: Path) -> None:
    args, protocol = _fixture(tmp_path)
    extra_root = Path(args.candidate[0][2]) / "extra"
    extra_root.mkdir()
    with pytest.raises(ValueError, match="candidate scene design mismatch"):
        _load_scenes(args, protocol)
