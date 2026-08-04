import json
from pathlib import Path

import pytest

from camcanon3r.prediction import PREDICTION_SCHEMA_VERSION
from scripts.summarize_prediction_compute import summarize_prediction_compute


def _write_metadata(path: Path, value: float, *, current_schema: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": ["000.png", "001.png", "002.png"],
        "load_seconds": 5.0,
        "inference_seconds": value,
        "peak_vram_bytes": 1024,
    }
    if current_schema:
        payload.update(
            {
                "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
                "model_load_seconds": 5.0,
                "model_compute_seconds": value,
                "end_to_end_seconds_excluding_model_load_and_metadata_write": (
                    value + 1.0
                ),
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_compute_requires_and_summarizes_complete_design(
    tmp_path: Path,
) -> None:
    root = tmp_path / "predictions"
    values = iter([1.0, 2.0, 3.0, 4.0])
    for scene in ("scan1", "scan2"):
        for variant in ("identity", "crop"):
            _write_metadata(
                root / scene / f"{variant}.json",
                next(values),
                current_schema=True,
            )
    report = summarize_prediction_compute(
        root,
        model="vggt",
        dataset="dtu-held-out",
        scenes=["scan1", "scan2"],
        variants=["identity", "crop"],
    )
    assert report["prediction_count"] == 4
    assert report["model_compute_seconds"]["median"] == 2.5
    assert report["model_compute_seconds"]["total"] == 10.0
    assert (
        report["end_to_end_seconds_excluding_model_load_and_metadata_write"]["total"]
        == 14.0
    )
    assert report["peak_vram_bytes"]["maximum"] == 1024

    _write_metadata(root / "extra" / "identity.json", 1.0, current_schema=True)
    with pytest.raises(ValueError, match="design mismatch"):
        summarize_prediction_compute(
            root,
            model="vggt",
            dataset="dtu-held-out",
            scenes=["scan1", "scan2"],
            variants=["identity", "crop"],
        )


def test_prediction_compute_preserves_legacy_end_to_end_unavailability(
    tmp_path: Path,
) -> None:
    root = tmp_path / "predictions"
    _write_metadata(root / "scene" / "identity.json", 2.0, current_schema=False)
    report = summarize_prediction_compute(
        root,
        model="vggt",
        dataset="eth3d",
        scenes=["scene"],
        variants=["identity"],
    )
    assert report["model_compute_seconds"]["total"] == 2.0
    assert report["end_to_end_available_count"] == 0
    assert (
        report["end_to_end_seconds_excluding_model_load_and_metadata_write"]["median"]
        is None
    )
    with pytest.raises(ValueError, match="end-to-end timing"):
        summarize_prediction_compute(
            root,
            model="vggt",
            dataset="eth3d",
            scenes=["scene"],
            variants=["identity"],
            require_end_to_end=True,
        )
