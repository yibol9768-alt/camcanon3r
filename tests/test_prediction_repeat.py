import json
from pathlib import Path

import numpy as np
import pytest

from camcanon3r.prediction_repeat import audit_prediction_repeat


def _prediction(root: Path, scene: str, *, delta: float = 0.0) -> None:
    path = root / scene / "identity.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        extrinsic=np.eye(4)[None],
        depth=np.array([[[1.0 + delta, np.nan]]]),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "inputs": ["one.png", "two.png"],
                "spatial_transforms": [{"input": "one.png"}],
                "seed": 17,
                "inference_seconds": 1.0 + delta,
                "load_seconds": 2.0,
                "peak_vram_bytes": 100,
            }
        ),
        encoding="utf-8",
    )


def test_prediction_repeat_reports_exact_protocol_arrays(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for scene in ("first", "second"):
        _prediction(reference, scene)
        _prediction(candidate, scene)
    report = audit_prediction_repeat(reference, candidate, scenes=["first", "second"])
    assert report["scene_count"] == 2
    assert report["exact_scene_count"] == 2
    assert report["array_count"] == 4
    assert report["exact_array_count"] == 4
    assert report["all_prediction_arrays_exact"] is True


def test_prediction_repeat_keeps_numeric_drift_visible(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _prediction(reference, "scene")
    _prediction(candidate, "scene", delta=0.25)
    report = audit_prediction_repeat(reference, candidate, scenes=["scene"])
    assert report["all_prediction_arrays_exact"] is False
    depth = next(
        record
        for record in report["records"][0]["arrays"]
        if record["field"] == "depth"
    )
    assert depth["maximum_absolute_difference"] == pytest.approx(0.25)


def test_prediction_repeat_rejects_protocol_metadata_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _prediction(reference, "scene")
    _prediction(candidate, "scene")
    metadata = candidate / "scene/identity.json"
    record = json.loads(metadata.read_text())
    record["seed"] = 18
    metadata.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol metadata changed"):
        audit_prediction_repeat(reference, candidate, scenes=["scene"])
