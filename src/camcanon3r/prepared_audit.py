"""Strict integrity audit for prepared transform sweeps."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .prediction import write_json_atomic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_crop_offset(
    record: Mapping[str, object], fraction: float
) -> tuple[float, float]:
    source_width, source_height = (int(value) for value in record["source_size"])
    matrix = np.asarray(record["matrix"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("prepared transform matrix must be finite and 3x3")
    expected_scale = 1.0 / fraction
    if not np.allclose(
        matrix,
        [
            [expected_scale, 0.0, matrix[0, 2]],
            [0.0, expected_scale, matrix[1, 2]],
            [0.0, 0.0, 1.0],
        ],
        atol=1e-12,
    ):
        raise ValueError("prepared crop matrix does not encode the named fraction")
    crop_x = -matrix[0, 2] / matrix[0, 0]
    crop_y = -matrix[1, 2] / matrix[1, 1]
    return (
        crop_x / (source_width * (1.0 - fraction)),
        crop_y / (source_height * (1.0 - fraction)),
    )


def audit_prepared_sweep(
    prepared_root: Path,
    variant_config_path: Path,
    scene_images: Mapping[str, Sequence[str]],
    *,
    reference_root: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    variants = [str(value) for value in config["ordered_variants"]]
    support_control = (
        config.get("experiment_role") == "support_preserving_coordinate_control"
    )
    variant_seeds = {
        str(key): int(value) for key, value in config["variant_seeds"].items()
    }
    if set(variant_seeds) != set(variants):
        raise ValueError("variant seed design does not match ordered variants")
    expected_scenes = set(scene_images)
    actual_scenes = {path.name for path in prepared_root.iterdir() if path.is_dir()}
    if actual_scenes != expected_scenes:
        raise ValueError(
            "prepared scene design mismatch: "
            f"missing={sorted(expected_scenes - actual_scenes)}, "
            f"extra={sorted(actual_scenes - expected_scenes)}"
        )

    reference_variants: list[str] = []
    if reference_root is not None:
        first_scene = min(expected_scenes)
        reference_manifest = json.loads(
            (reference_root / first_scene / "manifest.json").read_text(encoding="utf-8")
        )
        reference_variants = [
            str(record["name"]) for record in reference_manifest["variants"]
        ]
        if variants[: len(reference_variants)] != reference_variants:
            raise ValueError("reference variants are not a prefix of the sweep")

    tree_digest = hashlib.sha256()
    png_count = 0
    reference_match_count = 0
    shared_variant_count = 0
    support_content_matches = 0
    support_padding_matches = 0
    for scene in sorted(expected_scenes):
        scene_dir = prepared_root / scene
        manifest_path = scene_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("scene") != scene:
            raise ValueError(f"prepared manifest scene mismatch: {manifest_path}")
        records = manifest.get("variants")
        if not isinstance(records, list):
            raise TypeError(
                f"prepared manifest variants must be a list: {manifest_path}"
            )
        names = [str(record["name"]) for record in records]
        if names != variants:
            raise ValueError(f"prepared variant design mismatch: {manifest_path}")
        expected_sources = list(scene_images[scene])
        expected_outputs: set[Path] = {Path("manifest.json")}
        support_placements: dict[str, list[tuple[float, float]]] = {}

        reference_records: dict[str, object] = {}
        if reference_root is not None:
            reference_manifest_path = reference_root / scene / "manifest.json"
            reference_manifest = json.loads(
                reference_manifest_path.read_text(encoding="utf-8")
            )
            reference_records = {
                str(record["name"]): record for record in reference_manifest["variants"]
            }
            if list(reference_records) != reference_variants:
                raise ValueError(
                    f"reference variant order changed: {reference_manifest_path}"
                )

        for record in records:
            variant = str(record["name"])
            if int(record["seed"]) != variant_seeds[variant]:
                raise ValueError(f"prepared variant seed mismatch: {scene}/{variant}")
            images = record.get("images")
            if (
                not isinstance(images, list)
                or [str(image["source"]) for image in images] != expected_sources
            ):
                raise ValueError(f"prepared source design mismatch: {scene}/{variant}")
            if variant in reference_records and record != reference_records[variant]:
                raise ValueError(
                    f"prepared reference manifest changed: {scene}/{variant}"
                )

            crop_offsets: list[tuple[float, float]] = []
            if "crop_" in variant:
                fraction = int(variant.rsplit("_", 1)[1]) / 100.0
                crop_offsets = [
                    _normalized_crop_offset(image, fraction) for image in images
                ]
                if not all(
                    -1e-12 <= coordinate <= 1.0 + 1e-12
                    for offset in crop_offsets
                    for coordinate in offset
                ):
                    raise ValueError(
                        f"prepared crop window is outside the source: {scene}/{variant}"
                    )
            if variant.startswith("center_crop_") and not np.allclose(
                crop_offsets, (0.5, 0.5), atol=1e-12
            ):
                raise ValueError(f"center crop is not centered: {scene}/{variant}")
            if variant.startswith("shared_asymmetric_crop_"):
                if not np.allclose(crop_offsets, crop_offsets[0], atol=1e-12):
                    raise ValueError(
                        f"shared crop does not reuse one normalized window: "
                        f"{scene}/{variant}"
                    )
                shared_variant_count += 1

            if support_control:
                if variant not in {
                    "letterbox_square",
                    "shared_asymmetric_letterbox_square",
                    "asymmetric_letterbox_square",
                }:
                    raise ValueError(
                        f"unexpected support-control variant: {scene}/{variant}"
                    )
                support_placements[variant] = []

            for image_record in images:
                output = Path(str(image_record["output"]))
                expected_output = (
                    Path(variant) / f"{Path(str(image_record['source'])).stem}.png"
                )
                if output != expected_output:
                    raise ValueError(
                        f"prepared output path mismatch: {scene}/{variant}/{output}"
                    )
                expected_outputs.add(output)
                image_path = scene_dir / output
                with Image.open(image_path) as image:
                    expected_size = tuple(
                        int(value) for value in image_record["target_size"]
                    )
                    if image.size != expected_size or image.mode != "RGB":
                        raise ValueError(
                            f"prepared image shape/mode mismatch: {image_path}"
                        )
                    image.verify()
                if support_control:
                    source_path = Path(str(manifest["source_directory"])) / str(
                        image_record["source"]
                    )
                    if not source_path.is_file():
                        raise FileNotFoundError(
                            f"support-control source is missing: {source_path}"
                        )
                    source_width, source_height = (
                        int(value) for value in image_record["source_size"]
                    )
                    side = max(source_width, source_height)
                    matrix = np.asarray(image_record["matrix"], dtype=np.float64)
                    pad_x = float(matrix[0, 2])
                    pad_y = float(matrix[1, 2])
                    free_x = side - source_width
                    free_y = side - source_height
                    if tuple(int(value) for value in image_record["target_size"]) != (
                        side,
                        side,
                    ) or not np.allclose(
                        matrix,
                        [
                            [1.0, 0.0, pad_x],
                            [0.0, 1.0, pad_y],
                            [0.0, 0.0, 1.0],
                        ],
                        atol=1e-12,
                    ):
                        raise ValueError(
                            f"support control changes source scale: {scene}/{variant}"
                        )
                    expected_positions = (
                        {(free_x / 2.0, free_y / 2.0)}
                        if variant == "letterbox_square"
                        else {
                            (0.0, 0.0),
                            (float(free_x), 0.0),
                            (0.0, float(free_y)),
                        }
                    )
                    if (pad_x, pad_y) not in expected_positions:
                        raise ValueError(
                            f"support-control padding placement drift: "
                            f"{scene}/{variant}/{output}"
                        )
                    if not (pad_x.is_integer() and pad_y.is_integer()):
                        raise ValueError(
                            f"support control requires integer placement: "
                            f"{scene}/{variant}/{output}"
                        )
                    with Image.open(source_path) as source_image:
                        source_pixels = np.asarray(source_image.convert("RGB"))
                    with Image.open(image_path) as prepared_image:
                        prepared_pixels = np.asarray(prepared_image.convert("RGB"))
                    x0, y0 = int(pad_x), int(pad_y)
                    restored = prepared_pixels[
                        y0 : y0 + source_height, x0 : x0 + source_width
                    ]
                    if not np.array_equal(restored, source_pixels):
                        raise ValueError(
                            f"support-control RGB content changed: "
                            f"{scene}/{variant}/{output}"
                        )
                    padding_mask = np.ones((side, side), dtype=bool)
                    padding_mask[y0 : y0 + source_height, x0 : x0 + source_width] = (
                        False
                    )
                    if np.any(prepared_pixels[padding_mask] != 0):
                        raise ValueError(
                            f"support-control padding is not black: "
                            f"{scene}/{variant}/{output}"
                        )
                    support_content_matches += 1
                    support_padding_matches += 1
                    normalized = (
                        pad_x / free_x if free_x else 0.5,
                        pad_y / free_y if free_y else 0.5,
                    )
                    support_placements[variant].append(normalized)
                digest = _sha256(image_path)
                relative = image_path.relative_to(prepared_root).as_posix()
                tree_digest.update(relative.encode("utf-8") + b"\0")
                tree_digest.update(bytes.fromhex(digest))
                png_count += 1
                if variant in reference_records:
                    reference_path = reference_root / scene / output  # type: ignore[operator]
                    if digest != _sha256(reference_path):
                        raise ValueError(
                            f"prepared reference image changed: {image_path}"
                        )
                    reference_match_count += 1

        if support_control:
            shared = support_placements["shared_asymmetric_letterbox_square"]
            independent = support_placements["asymmetric_letterbox_square"]
            if not np.allclose(shared, shared[0], atol=1e-12):
                raise ValueError(
                    f"shared support-control placement differs across views: {scene}"
                )
            if len(set(independent)) < 2:
                raise ValueError(
                    f"independent support-control placement has no view variation: "
                    f"{scene}"
                )

        actual_outputs = {
            path.relative_to(scene_dir)
            for path in scene_dir.rglob("*")
            if path.is_file()
        }
        if actual_outputs != expected_outputs:
            raise ValueError(
                f"prepared file design mismatch for {scene}: "
                f"missing={sorted(str(path) for path in expected_outputs - actual_outputs)}, "
                f"extra={sorted(str(path) for path in actual_outputs - expected_outputs)}"
            )

    expected_png_count = sum(len(images) for images in scene_images.values()) * len(
        variants
    )
    if png_count != expected_png_count:
        raise RuntimeError("prepared audit counted an incomplete image design")
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "prepared_root": str(prepared_root.resolve()),
        "scene_count": len(scene_images),
        "variant_count": len(variants),
        "png_count": png_count,
        "manifest_count": len(scene_images),
        "tree_sha256": tree_digest.hexdigest(),
        "reference_root": str(reference_root.resolve()) if reference_root else None,
        "reference_variant_count": len(reference_variants),
        "reference_image_matches": reference_match_count,
        "shared_variant_scene_count": shared_variant_count,
        "support_control": support_control,
        "support_content_matches": support_content_matches,
        "support_padding_matches": support_padding_matches,
    }
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
