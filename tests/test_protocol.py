import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import camcanon3r.protocol as protocol_module
from camcanon3r.protocol import build_transform, prepare_scene, protocol_affines


def _write_scene(scene_dir: Path) -> None:
    scene_dir.mkdir()
    for index in range(3):
        array = np.zeros((40, 60, 3), dtype=np.uint8)
        array[..., 0] = 30 * index
        array[10:30, 20:40, 1] = 255
        Image.fromarray(array).save(scene_dir / f"view_{index}.png")


def test_prepare_scene_writes_exact_manifest_and_images(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    output = tmp_path / "prepared"
    _write_scene(scene)
    manifest = prepare_scene(
        scene,
        output,
        variants=("identity", "asymmetric_crop_075", "letterbox_square"),
        seed=29,
    )

    assert manifest["protocol_version"] == "0.1.0"
    assert len(manifest["variants"]) == 3
    assert all(len(variant["images"]) == 3 for variant in manifest["variants"])
    assert (output / "identity" / "view_0.png").stat().st_size > 0
    loaded = json.loads((output / "manifest.json").read_text())
    assert loaded == manifest


def test_asymmetric_crop_is_seed_deterministic(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_scene(scene)
    first = prepare_scene(
        scene,
        tmp_path / "first",
        variants=("asymmetric_crop_075",),
        seed=43,
    )
    second = prepare_scene(
        scene,
        tmp_path / "second",
        variants=("asymmetric_crop_075",),
        seed=43,
    )
    first_matrices = [image["matrix"] for image in first["variants"][0]["images"]]
    second_matrices = [image["matrix"] for image in second["variants"][0]["images"]]
    assert first_matrices == second_matrices


def test_shared_asymmetric_crop_reuses_one_window(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_scene(scene)
    manifest = prepare_scene(
        scene,
        tmp_path / "prepared",
        variants=("shared_asymmetric_crop_075", "asymmetric_crop_075"),
        seed=17,
    )

    shared = manifest["variants"][0]["images"]
    independent = manifest["variants"][1]["images"]
    np.testing.assert_allclose(shared[0]["matrix"], shared[1]["matrix"])
    assert not np.allclose(independent[0]["matrix"], independent[1]["matrix"])


def test_asymmetric_letterbox_preserves_support_and_freezes_scope(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    for index in range(4):
        array = np.full((40, 60, 3), 20 + index, dtype=np.uint8)
        array[5:35, 10:50, 1] = 100 + index
        Image.fromarray(array).save(scene / f"view_{index}.png")
    prepared = tmp_path / "prepared"
    manifest = prepare_scene(
        scene,
        prepared,
        variants=(
            "letterbox_square",
            "shared_asymmetric_letterbox_square",
            "asymmetric_letterbox_square",
        ),
        seed=17,
    )
    symmetric, shared, independent = manifest["variants"]
    assert {image["matrix"][1][2] for image in shared["images"]} == {0.0}
    assert {image["matrix"][1][2] for image in independent["images"]} == {
        0.0,
        20.0,
    }
    assert {image["matrix"][1][2] for image in symmetric["images"]} == {10.0}

    source = np.asarray(Image.open(scene / "view_0.png").convert("RGB"))
    for variant in manifest["variants"]:
        record = variant["images"][0]
        pad_y = int(record["matrix"][1][2])
        rendered = np.asarray(Image.open(prepared / record["output"]).convert("RGB"))
        np.testing.assert_array_equal(rendered[pad_y : pad_y + 40], source)


def test_crop_fraction_variants_encode_frozen_severities() -> None:
    import random

    rng = random.Random(17)
    center = build_transform("center_crop_060", (1000, 800), rng=rng)
    asymmetric = build_transform("asymmetric_crop_090", (1000, 800), rng=rng)
    assert center.matrix[0, 0] == pytest.approx(1 / 0.6)
    assert asymmetric.matrix[1, 1] == pytest.approx(1 / 0.9)
    with pytest.raises(ValueError, match="three-digit"):
        build_transform("center_crop_75", (1000, 800), rng=rng)


def test_protocol_affines_require_complete_variant_manifest(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    _write_scene(scene)
    prepared = tmp_path / "prepared"
    prepare_scene(scene, prepared, variants=["center_crop_075"], seed=17)
    variant = prepared / "center_crop_075"
    paths = sorted(variant.glob("*.png"))
    matrices = protocol_affines(variant, paths)
    assert len(matrices) == 3
    np.testing.assert_allclose(matrices[0][0, 0], 4 / 3)

    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["variants"][0]["images"].pop()
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="missing prepared inputs"):
        protocol_affines(variant, paths)


def test_prepare_scene_resume_skips_valid_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    output = tmp_path / "prepared"
    _write_scene(scene)
    expected = prepare_scene(
        scene, output, variants=("identity", "letterbox_square"), seed=17
    )

    def fail_if_rendered(*args: object, **kwargs: object) -> Image.Image:
        raise AssertionError("valid prepared output should have been reused")

    monkeypatch.setattr(protocol_module, "apply_affine", fail_if_rendered)
    resumed = prepare_scene(
        scene,
        output,
        variants=("identity", "letterbox_square"),
        seed=17,
        resume=True,
    )
    assert resumed == expected


def test_prepare_scene_resume_migrates_explicit_scene_name_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dslr_images"
    output = tmp_path / "courtyard"
    _write_scene(source)
    legacy = prepare_scene(source, output, variants=("identity",), seed=17)
    assert legacy["scene"] == "dslr_images"

    def fail_if_rendered(*args: object, **kwargs: object) -> Image.Image:
        raise AssertionError("scene-name migration must not rewrite images")

    monkeypatch.setattr(protocol_module, "apply_affine", fail_if_rendered)
    migrated = prepare_scene(
        source,
        output,
        variants=("identity",),
        seed=17,
        scene_name="courtyard",
        resume=True,
    )
    assert migrated["scene"] == "courtyard"
    assert json.loads((output / "manifest.json").read_text()) == migrated


def test_prepare_scene_resume_repairs_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene"
    output = tmp_path / "prepared"
    _write_scene(scene)
    prepare_scene(scene, output, variants=("identity",), seed=17)
    damaged = output / "identity" / "view_1.png"
    damaged.write_bytes(b"truncated")
    original_apply = protocol_module.apply_affine
    rendered_sources: list[tuple[int, int]] = []

    def record_render(
        image: Image.Image, transform: object, **kwargs: object
    ) -> Image.Image:
        rendered_sources.append(image.size)
        return original_apply(image, transform, **kwargs)

    monkeypatch.setattr(protocol_module, "apply_affine", record_render)
    prepare_scene(scene, output, variants=("identity",), seed=17, resume=True)

    assert rendered_sources == [(60, 40)]
    with Image.open(damaged) as repaired:
        repaired.verify()


def test_prepare_scene_resume_rejects_manifest_mismatch(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    output = tmp_path / "prepared"
    _write_scene(scene)
    prepare_scene(scene, output, variants=("asymmetric_crop_075",), seed=17)

    with pytest.raises(ValueError, match="manifest does not match"):
        prepare_scene(
            scene,
            output,
            variants=("asymmetric_crop_075",),
            seed=29,
            resume=True,
        )
