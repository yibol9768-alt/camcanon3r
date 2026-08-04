import json
from pathlib import Path

import pytest

from scripts.summarize_compute_table import summarize_compute_table


def _sweep(path: Path, *, model: str, dataset: str, end_count: int = 2) -> None:
    end = {
        "count": end_count,
        "median": 3.0 if end_count else None,
        "p90": 3.5 if end_count else None,
        "total": 6.0 if end_count else None,
    }
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "model": model,
                "dataset": dataset,
                "scene_count": 1,
                "variant_count": 2,
                "prediction_count": 2,
                "model_compute_seconds": {
                    "count": 2,
                    "median": 2.0,
                    "p90": 2.5,
                    "total": 4.0,
                },
                "end_to_end_seconds_excluding_model_load_and_metadata_write": end,
                "end_to_end_available_count": end_count,
                "model_load_seconds": {
                    "median": 5.0,
                    "minimum": 5.0,
                    "maximum": 5.0,
                },
                "peak_vram_bytes": {
                    "count": 2,
                    "median": 2**30,
                    "maximum": 2**31,
                },
                "records": [{"view_count": 3}, {"view_count": 3}],
            }
        ),
        encoding="utf-8",
    )


def test_compute_table_normalizes_complete_and_legacy_timings(tmp_path: Path) -> None:
    current = tmp_path / "current.json"
    legacy = tmp_path / "legacy.json"
    _sweep(current, model="vggt", dataset="dtu-held-out")
    _sweep(legacy, model="vggt", dataset="eth3d-training-raw", end_count=0)
    report = summarize_compute_table(
        [
            ("vggt", "dtu-held-out", "mechanism", current),
            ("vggt", "eth3d-training-raw", "mechanism", legacy),
        ]
    )
    assert report["sweep_count"] == 2
    assert report["sweeps"][0]["peak_vram_gibibytes"]["maximum"] == 2.0
    assert report["sweeps"][0]["end_to_end_availability"] == "complete"
    assert report["sweeps"][1]["end_to_end_availability"] == "legacy_unavailable"


def test_compute_table_rejects_partial_timing_and_duplicate_design(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "partial.json"
    _sweep(partial, model="vggt", dataset="dtu-held-out", end_count=1)
    with pytest.raises(ValueError, match="partially available"):
        summarize_compute_table([("vggt", "dtu-held-out", "mechanism", partial)])

    complete = tmp_path / "complete.json"
    _sweep(complete, model="vggt", dataset="dtu-held-out")
    with pytest.raises(ValueError, match="duplicate compute sweep"):
        summarize_compute_table(
            [
                ("vggt", "dtu-held-out", "mechanism", complete),
                ("vggt", "dtu-held-out", "mechanism", complete),
            ]
        )


def test_compute_table_binds_canonicalization_wall_time(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep.json"
    _sweep(sweep, model="vggt", dataset="dtu-held-out")
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "status": "complete",
                "record_count": 2,
                "fill_policy": "neutral_gray",
                "timing_boundary": "decode, warp, and write",
                "canonicalization_seconds": {"count": 2, "total": 3.0},
                "records": [
                    {
                        "scene": "scan1",
                        "source_variant": "identity",
                        "canonicalization_seconds": 1.0,
                    },
                    {
                        "scene": "scan1",
                        "source_variant": "asymmetric_crop_075",
                        "canonicalization_seconds": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report = summarize_compute_table(
        [("vggt", "dtu-held-out", "mechanism", sweep)],
        canonicalization=canonical,
    )
    assert report["canonicalization"]["fill_policy"] == "neutral_gray"
    assert (
        report["canonicalization"]["by_source_variant"]["asymmetric_crop_075"][
            "median_seconds"
        ]
        == 2.0
    )
