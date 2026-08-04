import json
from pathlib import Path

import pytest

from scripts.summarize_cross_dataset_table import VARIANTS, summarize_one


def _interval(value: float) -> dict[str, float]:
    return {"point_estimate": value, "lower": value - 0.1, "upper": value + 0.1}


def _summary(schema: str) -> dict[str, object]:
    by_variant = {}
    for variant in VARIANTS:
        delta = 0.0 if variant == "identity" else 1.0
        metrics = {
            "rotation_median_degrees": _interval(2.0 + delta),
            "translation_median_degrees": _interval(3.0 + delta),
            "focal_relative_error_median": _interval(0.02 + delta / 100),
            "principal_point_normalized_error_median": _interval(0.03 + delta / 100),
        }
        if schema == "eth3d":
            metrics.update(
                {
                    "rotation_delta_from_identity_degrees": _interval(delta),
                    "translation_delta_from_identity_degrees": _interval(delta),
                    "focal_relative_error_delta_from_identity": _interval(delta / 100),
                    "principal_point_normalized_error_delta_from_identity": (
                        _interval(delta / 100)
                    ),
                    "depth_abs_rel_delta_from_identity": _interval(delta / 100),
                    "point_accuracy_delta_from_identity_meters": _interval(
                        delta / 1000
                    ),
                    "point_completeness_delta_from_identity_meters": _interval(
                        delta / 1000
                    ),
                }
            )
        else:
            metrics.update(
                {
                    "rotation_median_degrees_delta_from_identity": _interval(delta),
                    "translation_median_degrees_delta_from_identity": _interval(delta),
                    "focal_relative_error_median_delta_from_identity": _interval(
                        delta / 100
                    ),
                    "principal_point_normalized_error_median_delta_from_identity": (
                        _interval(delta / 100)
                    ),
                    "point_accuracy_mean_millimeters_delta_from_identity": (
                        _interval(delta)
                    ),
                    "point_completeness_mean_millimeters_delta_from_identity": (
                        _interval(delta)
                    ),
                }
            )
        by_variant[variant] = {
            "scene_count": 2,
            "scene_bootstrap": {"metrics": metrics},
        }
    result = {"scene_count": 2, "by_variant": by_variant}
    if schema == "eth3d":
        result["depth_evaluated"] = True
    else:
        result["point_metric_protocol"] = "test"
    return result


@pytest.mark.parametrize("schema", ["eth3d", "dtu"])
def test_cross_dataset_table_normalizes_units(tmp_path: Path, schema: str) -> None:
    path = tmp_path / f"{schema}.json"
    path.write_text(json.dumps(_summary(schema)), encoding="utf-8")
    result = summarize_one("model", schema, path)
    candidate = next(
        record
        for record in result["variants"]
        if record["variant"] == "asymmetric_crop_075"
    )
    delta = candidate["delta_from_identity"]
    assert delta["rotation_degrees"]["point_estimate"] == 1.0
    assert delta["focal_relative"]["point_estimate"] == 1.0
    assert delta["principal_point_normalized"]["point_estimate"] == 1.0
    assert delta["point_accuracy_millimeters"]["point_estimate"] == 1.0
    assert (delta["depth_abs_rel"] is None) == (schema == "dtu")
