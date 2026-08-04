from __future__ import annotations

import json
from pathlib import Path

import pytest

from camcanon3r.support_analysis import analyze_support_control
from camcanon3r.support_control import SUPPORT_VARIANTS


def _summary(path: Path, dataset: str, scene_count: int) -> None:
    rows = []
    offsets = {
        "letterbox_square": 0.0,
        "shared_asymmetric_letterbox_square": 1.0,
        "asymmetric_letterbox_square": 3.0,
    }
    for scene_index in range(scene_count):
        for variant in reversed(SUPPORT_VARIANTS):
            offset = offsets[variant]
            row = {
                "scene": f"scene{scene_index:02d}",
                "variant": variant,
                "rotation_median_degrees": 0.5 + offset,
                "translation_median_degrees": 1.0 + offset,
                "focal_relative_error_median": 0.01 + offset / 100.0,
                "principal_point_normalized_error_median": 0.02 + offset / 100.0,
            }
            if dataset == "eth3d":
                row.update(
                    {
                        "depth_mean_abs_rel": 0.03 + offset / 100.0,
                        "point_accuracy_mean_meters": 0.1 + offset / 100.0,
                        "point_completeness_mean_meters": 0.2 + offset / 100.0,
                    }
                )
            else:
                row.update(
                    {
                        "point_accuracy_mean_millimeters": 1.0 + offset,
                        "point_completeness_mean_millimeters": 2.0 + offset,
                    }
                )
            rows.append(row)
    payload = {
        "scene_count": scene_count,
        "evaluations": rows,
    }
    if dataset == "eth3d":
        payload["depth_evaluated"] = True
    else:
        payload["point_metric_protocol"] = "frozen"
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_support_analysis_requires_complete_cross_dataset_design(
    tmp_path: Path,
) -> None:
    inputs = []
    for model in ("vggt", "dust3r"):
        for dataset, scenes in (("eth3d", 13), ("dtu", 22)):
            path = tmp_path / f"{model}_{dataset}.json"
            _summary(path, dataset, scenes)
            inputs.append((model, dataset, path))
    report = analyze_support_control(
        inputs,
        Path("configs/support_control_variants.json"),
        bootstrap_replicates=100,
        bootstrap_seed=17,
    )
    assert report["status"] == "complete"
    assert report["promotion_gate"]["passes_all_models_and_datasets"] is True
    assert all(
        record["rotation_delta"]["point_estimate"] == pytest.approx(3.0)
        for record in report["promotion_gate"]["support"]
    )

    with pytest.raises(ValueError, match="model/dataset design"):
        analyze_support_control(
            inputs[:-1],
            Path("configs/support_control_variants.json"),
            bootstrap_replicates=10,
        )
