import hashlib
import json
from pathlib import Path

import pytest

from camcanon3r.mechanism_analysis import analyze_mechanism_summaries

VARIANTS = (
    "identity",
    "center_crop_075",
    "asymmetric_crop_075",
    "letterbox_square",
    "center_crop_090",
    "center_crop_060",
    "asymmetric_crop_090",
    "asymmetric_crop_060",
    "shared_asymmetric_crop_090",
    "shared_asymmetric_crop_075",
    "shared_asymmetric_crop_060",
)


def _write_config(path: Path, variants: tuple[str, ...] = VARIANTS) -> None:
    path.write_text(
        json.dumps(
            {
                "frozen_before_benchmark_scale_mechanism_results": True,
                "ordered_variants": list(variants),
            }
        ),
        encoding="utf-8",
    )


def _write_summary(path: Path, *, center_crosses: bool = True) -> None:
    rotation_deltas = {
        "identity": 0.0,
        "center_crop_090": 0.5,
        "center_crop_075": 1.0,
        "center_crop_060": 2.5 if center_crosses else 1.5,
        "asymmetric_crop_090": 1.0,
        "asymmetric_crop_075": 3.0,
        "asymmetric_crop_060": 5.0,
        "shared_asymmetric_crop_090": 0.1,
        "shared_asymmetric_crop_075": 0.2,
        "shared_asymmetric_crop_060": 0.3,
        "letterbox_square": 0.1,
    }
    evaluations = []
    for scene_index, scene in enumerate(("first", "second", "third")):
        identity_rotation = 1.0 + scene_index * 0.1
        for variant in VARIANTS:
            rotation = identity_rotation + rotation_deltas[variant]
            evaluations.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "rotation_median_degrees": rotation,
                    "translation_median_degrees": rotation * 2.0,
                    "focal_relative_error_median": rotation / 100.0,
                    "principal_point_normalized_error_median": rotation / 200.0,
                    "depth_mean_abs_rel": 0.1 + rotation / 100.0,
                }
            )
    path.write_text(json.dumps({"evaluations": evaluations}), encoding="utf-8")


def test_mechanism_analysis_is_paired_and_enforces_cross_dataset_gate(
    tmp_path: Path,
) -> None:
    config = tmp_path / "variants.json"
    _write_config(config)
    records = []
    for model in ("vggt", "dust3r"):
        for dataset in ("eth3d", "dtu"):
            summary = tmp_path / f"{model}_{dataset}.json"
            _write_summary(summary)
            records.append((model, dataset, summary))

    report = analyze_mechanism_summaries(
        records,
        config,
        bootstrap_replicates=100,
    )

    eth_vggt = report["analyses"]["vggt/eth3d"]["analysis"]
    assert report["analyses"]["vggt/eth3d"]["summary"] == str(
        tmp_path / "vggt_eth3d.json"
    )
    assert report["analyses"]["vggt/eth3d"]["summary_sha256"] == hashlib.sha256(
        (tmp_path / "vggt_eth3d.json").read_bytes()
    ).hexdigest()
    assert report["variant_config"] == str(config)
    assert report["variant_config_sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    asymmetric = eth_vggt["family_gates"]["independent_asymmetric_crop"]
    assert asymmetric["rotation_delta_point_estimates"] == pytest.approx(
        [1.0, 3.0, 5.0]
    )
    assert asymmetric["rotation_monotone_as_retention_decreases"] is True
    contrast = eth_vggt["paired_contrasts"]["independent_minus_shared_075"]
    assert contrast["metrics"][
        "rotation_median_degrees_delta_from_identity"
    ]["point_estimate"] == pytest.approx(2.8)
    assert report["family_support"]["independent_asymmetric_crop"][
        "meets_two_dataset_gate"
    ] is True
    assert report["family_support"]["center_crop"][
        "datasets_with_all_evaluated_models_crossing"
    ] == ["dtu", "eth3d"]
    assert report["hypothesis_gate"]["meets_two_family_two_dataset_gate"] is True


def test_mechanism_analysis_requires_every_model_for_dataset_support(
    tmp_path: Path,
) -> None:
    config = tmp_path / "variants.json"
    _write_config(config)
    records = []
    for model, center_crosses in (("vggt", True), ("dust3r", False)):
        summary = tmp_path / f"{model}.json"
        _write_summary(summary, center_crosses=center_crosses)
        records.append((model, "eth3d", summary))

    report = analyze_mechanism_summaries(
        records,
        config,
        bootstrap_replicates=20,
    )
    support = report["family_support"]["center_crop"]
    assert support["crossing_models_by_dataset"]["eth3d"] == ["vggt"]
    assert support["datasets_with_all_evaluated_models_crossing"] == []
    assert support["meets_two_dataset_gate"] is False


def test_mechanism_analysis_rejects_duplicate_or_unfrozen_variant_design(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    _write_summary(summary)
    duplicate_config = tmp_path / "duplicate.json"
    _write_config(duplicate_config, VARIANTS + ("identity",))
    with pytest.raises(ValueError, match="does not match frozen families"):
        analyze_mechanism_summaries(
            [("vggt", "eth3d", summary)], duplicate_config
        )

    unfrozen = tmp_path / "unfrozen.json"
    unfrozen.write_text(json.dumps({"ordered_variants": list(VARIANTS)}))
    with pytest.raises(ValueError, match="is not frozen"):
        analyze_mechanism_summaries([("vggt", "eth3d", summary)], unfrozen)


def test_mechanism_analysis_keeps_partial_secondary_metric_explicit(
    tmp_path: Path,
) -> None:
    config = tmp_path / "variants.json"
    summary = tmp_path / "summary.json"
    _write_config(config)
    _write_summary(summary)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    target = next(
        row
        for row in payload["evaluations"]
        if row["scene"] == "second" and row["variant"] == "center_crop_075"
    )
    target["translation_median_degrees"] = None
    summary.write_text(json.dumps(payload), encoding="utf-8")

    report = analyze_mechanism_summaries(
        [("vggt", "eth3d", summary)],
        config,
        bootstrap_replicates=20,
    )
    variant = report["analyses"]["vggt/eth3d"]["analysis"]["by_variant"][
        "center_crop_075"
    ]
    availability = variant["metric_availability"][
        "translation_median_degrees"
    ]
    assert availability["paired_valid_scene_count"] == 2
    assert availability["undefined_scene_count"] == 1
    assert availability["included_in_scene_bootstrap"] is False
    assert "translation_median_degrees" not in variant["metrics"]
    assert "rotation_median_degrees" in variant["metrics"]
