import json
from pathlib import Path

from camcanon3r.summary import summarize_comparison_files


def _write_record(path: Path, *, scene: str, candidate: str, rotation: float) -> None:
    path.write_text(
        json.dumps(
            {
                "reference": f"/outputs/{scene}/identity.npz",
                "candidate_label": candidate,
                "rotation_degrees": {"median": rotation},
                "translation_direction_degrees": {"median": rotation * 2},
                "aligned_depth_consistency": {
                    "mean_abs_rel": rotation / 100,
                    "valid_pixels": 1000,
                },
            }
        ),
        encoding="utf-8",
    )


def test_summary_counts_thresholds_by_variant(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_record(first, scene="room", candidate="asymmetric_crop_075", rotation=3.0)
    _write_record(second, scene="fern", candidate="asymmetric_crop_075", rotation=1.0)
    summary = summarize_comparison_files([first, second])
    aggregate = summary["by_variant"]["asymmetric_crop_075"]
    assert aggregate["scene_count"] == 2
    assert aggregate["scenes_over_rotation_threshold"] == 1
    assert aggregate["median_of_scene_rotation_medians_degrees"] == 2.0
