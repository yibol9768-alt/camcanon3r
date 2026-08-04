import json
from pathlib import Path


def test_mechanism_variant_seed_schedule_is_order_frozen() -> None:
    design = json.loads(
        Path("configs/eth3d_mechanism_variants.json").read_text(encoding="utf-8")
    )
    variants = design["ordered_variants"]
    seeds = design["variant_seeds"]
    base_seed = design["base_seed"]
    stride = design["variant_seed_stride"]

    assert variants[:4] == [
        "identity",
        "center_crop_075",
        "asymmetric_crop_075",
        "letterbox_square",
    ]
    assert set(seeds) == set(variants)
    assert [seeds[name] for name in variants] == [
        base_seed + stride * index for index in range(len(variants))
    ]


def test_support_control_is_frozen_before_dtu_ground_truth() -> None:
    design = json.loads(
        Path("configs/support_control_variants.json").read_text(encoding="utf-8")
    )
    variants = design["ordered_variants"]
    assert design["frozen_before_dtu_gt_inspection"] is True
    assert design["frozen_before_benchmark_scale_mechanism_results"] is False
    assert design["registered_after_eth3d_mechanism_results"] is True
    assert design["frozen_before_support_control_results"] is True
    assert variants == [
        "letterbox_square",
        "shared_asymmetric_letterbox_square",
        "asymmetric_letterbox_square",
    ]
    assert [design["variant_seeds"][name] for name in variants] == [
        design["base_seed"] + design["variant_seed_stride"] * index
        for index in range(len(variants))
    ]

    dtu = json.loads(
        Path("configs/dtu_support_control_protocol.json").read_text(encoding="utf-8")
    )
    base = json.loads(Path("configs/dtu_protocol.json").read_text(encoding="utf-8"))
    assert dtu["frozen_before_dtu_gt_inspection"] is True
    for field in (
        "evaluation_scans",
        "view_indices_zero_based",
        "rectified_archive_camera_ids_one_based",
        "lighting_index",
    ):
        assert dtu[field] == base[field]
    assert dtu["confirmatory_variants"] == variants
