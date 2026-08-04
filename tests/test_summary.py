import json
from pathlib import Path

import pytest

from camcanon3r.summary import (
    summarize_comparison_files,
    summarize_eth3d_evaluations,
)


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
    bootstrap = aggregate["scene_bootstrap"]
    assert bootstrap["resampling_unit"] == "scene"
    assert bootstrap["small_sample_warning"].startswith("descriptive_only")
    assert bootstrap["metrics"]["rotation_median_degrees"][
        "point_estimate"
    ] == 2.0


def test_summary_keeps_undefined_scene_metric_explicit(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_record(first, scene="room", candidate="crop", rotation=3.0)
    _write_record(second, scene="fern", candidate="crop", rotation=5.0)
    record = json.loads(second.read_text())
    record["translation_direction_degrees"]["median"] = None
    second.write_text(json.dumps(record))

    summary = summarize_comparison_files([first, second])
    crop = summary["by_variant"]["crop"]
    availability = crop["metric_availability"][
        "translation_median_degrees"
    ]
    assert crop["median_of_scene_translation_medians_degrees"] is None
    assert availability == {
        "valid_scene_count": 1,
        "undefined_scene_count": 1,
        "included_in_scene_bootstrap": False,
    }
    assert "translation_median_degrees" not in crop["scene_bootstrap"][
        "metrics"
    ]
    assert crop["scene_bootstrap"]["metrics"]["rotation_median_degrees"][
        "point_estimate"
    ] == 4.0


def test_eth3d_summary_reports_deltas_from_identity(tmp_path: Path) -> None:
    for variant, rotation, translation, depth in (
        ("identity", 1.0, 2.0, 0.10),
        ("crop", 3.5, 8.0, 0.16),
    ):
        path = tmp_path / f"{variant}_vs_gt.json"
        path.write_text(
            json.dumps(
                {
                    "prediction": f"/outputs/{variant}.npz",
                    "variant": variant,
                    "intrinsics": {
                        "focal_relative_error": {"median": rotation / 100},
                        "principal_point_normalized_error": {
                            "median": rotation / 200
                        },
                    },
                    "relative_rotation_degrees": {"median": rotation},
                    "translation_direction_degrees": {"median": translation},
                    "depth": {
                        "mean_abs_rel": depth,
                        "valid_pixels": 100,
                    },
                    "point_cloud": {
                        "accuracy_meters": {"mean": rotation / 10},
                        "completeness_meters": {"mean": translation / 10},
                    },
                }
            )
        )
    summary = summarize_eth3d_evaluations(
        sorted(tmp_path.glob("*_vs_gt.json"))
    )
    crop = next(row for row in summary["evaluations"] if row["variant"] == "crop")
    assert crop["rotation_delta_from_identity_degrees"] == 2.5
    assert crop["translation_delta_from_identity_degrees"] == 6.0
    assert crop["focal_relative_error_delta_from_identity"] == pytest.approx(
        0.025
    )
    assert crop[
        "principal_point_normalized_error_delta_from_identity"
    ] == pytest.approx(0.0125)
    assert crop["depth_abs_rel_delta_from_identity"] == pytest.approx(0.06)
    assert crop["point_accuracy_delta_from_identity_meters"] == pytest.approx(
        0.25
    )
    assert crop[
        "point_completeness_delta_from_identity_meters"
    ] == pytest.approx(0.6)
    assert summary["point_cloud_evaluated"] is True
    assert summary["by_variant"]["crop"]["scene_bootstrap"]["metrics"][
        "point_accuracy_mean_meters"
    ]["point_estimate"] == pytest.approx(0.35)
    assert summary["scene_count"] == 1
    assert summary["by_variant"]["crop"]["scene_bootstrap"]["scene_count"] == 1


def test_eth3d_summary_pairs_identity_within_each_scene(tmp_path: Path) -> None:
    for scene, identity_rotation, crop_rotation in (
        ("office", 1.0, 4.0),
        ("courtyard", 2.0, 7.0),
    ):
        scene_dir = tmp_path / scene
        scene_dir.mkdir()
        for variant, rotation in (
            ("identity", identity_rotation),
            ("crop", crop_rotation),
        ):
            (scene_dir / f"{variant}_vs_gt.json").write_text(
                json.dumps(
                    {
                        "prediction": f"/outputs/{scene}/{variant}.npz",
                        "variant": variant,
                        "intrinsics": {
                            "focal_relative_error": {"median": rotation / 100},
                            "principal_point_normalized_error": {
                                "median": rotation / 200
                            },
                        },
                        "relative_rotation_degrees": {"median": rotation},
                        "translation_direction_degrees": {"median": rotation * 2},
                        "depth": {"mean_abs_rel": rotation / 100, "valid_pixels": 10},
                    }
                )
            )

    summary = summarize_eth3d_evaluations(
        sorted(tmp_path.rglob("*_vs_gt.json")), bootstrap_replicates=200
    )
    assert summary["scene_count"] == 2
    assert summary["identity"] is None
    crop_rows = [
        row for row in summary["evaluations"] if row["variant"] == "crop"
    ]
    assert [row["rotation_delta_from_identity_degrees"] for row in crop_rows] == [
        5.0,
        3.0,
    ]
    crop_bootstrap = summary["by_variant"]["crop"]["scene_bootstrap"]
    assert crop_bootstrap["metrics"]["rotation_delta_from_identity_degrees"][
        "point_estimate"
    ] == 4.0


def test_eth3d_summary_keeps_undefined_point_alignment_explicit(
    tmp_path: Path,
) -> None:
    for variant, point_mean in (("identity", 0.1), ("crop", None)):
        (tmp_path / f"{variant}_vs_gt.json").write_text(
            json.dumps(
                {
                    "prediction": f"/outputs/{variant}.npz",
                    "variant": variant,
                    "intrinsics": {
                        "focal_relative_error": {"median": 0.1},
                        "principal_point_normalized_error": {"median": 0.01},
                    },
                    "relative_rotation_degrees": {"median": 1.0},
                    "translation_direction_degrees": {"median": 2.0},
                    "depth": {"mean_abs_rel": 0.1, "valid_pixels": 10},
                    "point_cloud": {
                        "accuracy_meters": {"mean": point_mean},
                        "completeness_meters": {"mean": point_mean},
                    },
                }
            )
        )
    summary = summarize_eth3d_evaluations(
        sorted(tmp_path.glob("*_vs_gt.json"))
    )
    availability = summary["by_variant"]["crop"]["metric_availability"][
        "point_accuracy_mean_meters"
    ]
    assert summary["point_cloud_evaluated"] is True
    assert availability["undefined_scene_count"] == 1
    assert availability["included_in_scene_bootstrap"] is False


def test_eth3d_summary_rejects_duplicate_scene_variant(tmp_path: Path) -> None:
    scene = tmp_path / "office"
    scene.mkdir()
    record = {
        "prediction": "/outputs/identity.npz",
        "variant": "identity",
        "intrinsics": {
            "focal_relative_error": {"median": 0.1},
            "principal_point_normalized_error": {"median": 0.01},
        },
        "relative_rotation_degrees": {"median": 1.0},
        "translation_direction_degrees": {"median": 2.0},
        "depth": None,
    }
    first = scene / "first_vs_gt.json"
    second = scene / "second_vs_gt.json"
    first.write_text(json.dumps(record))
    second.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duplicate ETH3D scene/variant"):
        summarize_eth3d_evaluations([first, second])


def test_eth3d_summary_rejects_incomplete_paired_design(tmp_path: Path) -> None:
    paths = []
    for scene, variants in (
        ("courtyard", ("identity", "crop")),
        ("office", ("identity",)),
    ):
        scene_dir = tmp_path / scene
        scene_dir.mkdir()
        for variant in variants:
            path = scene_dir / f"{variant}_vs_gt.json"
            path.write_text(
                json.dumps(
                    {
                        "prediction": f"/outputs/{scene}/{variant}.npz",
                        "variant": variant,
                        "intrinsics": {
                            "focal_relative_error": {"median": 0.1},
                            "principal_point_normalized_error": {
                                "median": 0.01
                            },
                        },
                        "relative_rotation_degrees": {"median": 1.0},
                        "translation_direction_degrees": {"median": 2.0},
                        "depth": None,
                    }
                )
            )
            paths.append(path)
    with pytest.raises(ValueError, match="incomplete paired ETH3D design"):
        summarize_eth3d_evaluations(paths)


def test_eth3d_summary_preserves_undefined_pose_metric(tmp_path: Path) -> None:
    paths = []
    for scene in ("office", "courtyard"):
        scene_dir = tmp_path / scene
        scene_dir.mkdir()
        for variant in ("identity", "crop"):
            translation = None if scene == "courtyard" and variant == "crop" else 2.0
            path = scene_dir / f"{variant}_vs_gt.json"
            path.write_text(
                json.dumps(
                    {
                        "prediction": f"/outputs/{scene}/{variant}.npz",
                        "variant": variant,
                        "intrinsics": {
                            "focal_relative_error": {"median": 0.1},
                            "principal_point_normalized_error": {
                                "median": 0.01
                            },
                        },
                        "relative_rotation_degrees": {"median": 1.0},
                        "translation_direction_degrees": {
                            "median": translation
                        },
                        "depth": None,
                    }
                )
            )
            paths.append(path)

    summary = summarize_eth3d_evaluations(paths)
    crop = summary["by_variant"]["crop"]
    availability = crop["metric_availability"][
        "translation_median_degrees"
    ]
    assert availability["valid_scene_count"] == 1
    assert availability["undefined_scene_count"] == 1
    assert availability["included_in_scene_bootstrap"] is False
    assert "translation_median_degrees" not in crop["scene_bootstrap"][
        "metrics"
    ]
