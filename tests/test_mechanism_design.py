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
