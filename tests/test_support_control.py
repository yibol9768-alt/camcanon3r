from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from camcanon3r.protocol import prepare_scene
from camcanon3r.support_control import SUPPORT_VARIANTS, audit_support_control


def _source(path: Path) -> list[str]:
    path.mkdir()
    names = []
    for index in range(3):
        name = f"view_{index}.png"
        Image.new("RGB", (12, 8), color=(30 + index, 60, 90)).save(path / name)
        names.append(name)
    return names


def test_support_control_audit_binds_main_letterbox_anchor(tmp_path: Path) -> None:
    source = tmp_path / "source"
    names = _source(source)
    support = tmp_path / "support"
    reference = tmp_path / "reference"
    prepare_scene(
        source,
        support / "scene",
        variants=SUPPORT_VARIANTS,
        seed=17,
        scene_name="scene",
    )
    prepare_scene(
        source,
        reference / "scene",
        variants=(
            "identity",
            "center_crop_075",
            "asymmetric_crop_075",
            "letterbox_square",
        ),
        seed=17,
        scene_name="scene",
    )
    report = audit_support_control(
        support,
        reference,
        Path("configs/support_control_variants.json"),
        {"scene": names},
        output_path=tmp_path / "audit.json",
    )
    assert report["status"] == "complete"
    assert report["reference_letterbox_matches"] == 3
    assert report["support_content_matches"] == 9

    Image.new("RGB", (12, 12), color=(1, 2, 3)).save(
        reference / "scene/letterbox_square/view_0.png"
    )
    with pytest.raises(ValueError, match="anchor image drift"):
        audit_support_control(
            support,
            reference,
            Path("configs/support_control_variants.json"),
            {"scene": names},
        )
