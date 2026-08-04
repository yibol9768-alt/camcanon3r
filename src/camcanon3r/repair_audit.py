"""Integrity audit for canonical-camera repair inputs and validity masks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
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


def _read_verified(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        return np.asarray(opened.convert(mode))


def _output_variant(source_variant: str, prefix: str) -> str:
    return "identity" if source_variant == "identity" else f"{prefix}{source_variant}"


def audit_canonical_repairs(
    prepared_root: Path,
    repaired_root: Path,
    *,
    scenes: Sequence[str],
    source_variants: Sequence[str],
    prefix: str = "canonical_",
    fill_policy: str | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    scene_names = [str(value) for value in scenes]
    variant_names = [str(value) for value in source_variants]
    if not scene_names or len(set(scene_names)) != len(scene_names):
        raise ValueError("repair audit scenes must be non-empty and unique")
    if not variant_names or len(set(variant_names)) != len(variant_names):
        raise ValueError("repair audit variants must be non-empty and unique")
    expected_scene_set = set(scene_names)
    actual_scene_set = {path.name for path in repaired_root.iterdir() if path.is_dir()}
    if actual_scene_set != expected_scene_set:
        raise ValueError(
            "repair scene design mismatch: "
            f"missing={sorted(expected_scene_set - actual_scene_set)}, "
            f"extra={sorted(actual_scene_set - expected_scene_set)}"
        )

    tree_digest = hashlib.sha256()
    image_count = 0
    mask_count = 0
    identity_pixel_matches = 0
    valid_fractions: dict[str, list[float]] = {
        _output_variant(variant, prefix): [] for variant in variant_names
    }
    for scene in scene_names:
        source_manifest_path = prepared_root / scene / "manifest.json"
        repaired_manifest_path = repaired_root / scene / "manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        repaired_manifest = json.loads(
            repaired_manifest_path.read_text(encoding="utf-8")
        )
        if (
            source_manifest.get("scene") != scene
            or repaired_manifest.get("scene") != scene
        ):
            raise ValueError(f"repair manifest scene mismatch: {scene}")
        source_records = {
            str(record["name"]): record for record in source_manifest["variants"]
        }
        repaired_records = {
            str(record["name"]): record for record in repaired_manifest["variants"]
        }
        expected_output_variants = [
            _output_variant(variant, prefix) for variant in variant_names
        ]
        if list(repaired_records) != expected_output_variants:
            raise ValueError(
                f"repair variant design mismatch for {scene}: "
                f"expected={expected_output_variants}, "
                f"actual={list(repaired_records)}"
            )
        expected_files = {Path("manifest.json")}
        for source_variant, output_variant in zip(
            variant_names, expected_output_variants, strict=True
        ):
            if source_variant not in source_records:
                raise ValueError(
                    f"source repair variant is missing: {scene}/{source_variant}"
                )
            source_images = source_records[source_variant]["images"]
            repaired_images = repaired_records[output_variant]["images"]
            if len(source_images) != len(repaired_images) or not source_images:
                raise ValueError(
                    f"repair image design mismatch: {scene}/{output_variant}"
                )
            for source_record, repaired_record in zip(
                source_images, repaired_images, strict=True
            ):
                repair = repaired_record.get("repair")
                if not isinstance(repair, dict):
                    raise TypeError(
                        f"repair provenance is missing: {scene}/{output_variant}"
                    )
                if (
                    repair.get("method") != "known_affine_inverse_canvas"
                    or repair.get("input_variant") != source_variant
                    or repair.get("input_affine") != source_record["matrix"]
                    or repaired_record.get("source") != source_record["source"]
                    or repaired_record.get("matrix") != np.eye(3).tolist()
                    or repaired_record.get("source_size")
                    != source_record["source_size"]
                    or repaired_record.get("target_size")
                    != source_record["source_size"]
                ):
                    raise ValueError(
                        f"repair provenance mismatch: {scene}/{output_variant}"
                    )
                if fill_policy is not None and repair.get("fill_policy") != fill_policy:
                    raise ValueError(
                        f"repair fill policy mismatch for {scene}/{output_variant}: "
                        f"expected={fill_policy!r}, "
                        f"actual={repair.get('fill_policy')!r}"
                    )
                output = Path(str(repaired_record["output"]))
                mask = Path(str(repair["valid_mask"]))
                expected_name = Path(str(source_record["output"])).name
                if output != Path(output_variant) / expected_name:
                    raise ValueError(
                        f"repair output path mismatch: {scene}/{output_variant}"
                    )
                if mask != Path("_masks") / output_variant / expected_name:
                    raise ValueError(
                        f"repair mask path mismatch: {scene}/{output_variant}"
                    )
                expected_files.update((output, mask))
                output_path_full = repaired_root / scene / output
                mask_path_full = repaired_root / scene / mask
                source_path = prepared_root / scene / source_record["output"]
                repaired_pixels = _read_verified(output_path_full, "RGB")
                mask_pixels = _read_verified(mask_path_full, "L")
                expected_size = tuple(
                    int(value) for value in source_record["source_size"]
                )
                if repaired_pixels.shape[:2] != expected_size[::-1]:
                    raise ValueError(f"repair image size mismatch: {output_path_full}")
                if (
                    mask_pixels.shape != expected_size[::-1]
                    or not np.isin(mask_pixels, (0, 255)).all()
                ):
                    raise ValueError(
                        f"repair validity mask is invalid: {mask_path_full}"
                    )
                valid_fraction = float(mask_pixels.mean() / 255.0)
                if not np.isclose(
                    valid_fraction, float(repair["valid_fraction"]), atol=1e-12
                ):
                    raise ValueError(
                        f"repair valid fraction mismatch: {scene}/{output_variant}"
                    )
                valid_fractions[output_variant].append(valid_fraction)
                fill = np.asarray(repair["fill_rgb"], dtype=np.uint8)
                if np.any(mask_pixels == 0) and not np.all(
                    repaired_pixels[mask_pixels == 0] == fill
                ):
                    raise ValueError(
                        f"repair pixels outside valid support do not match fill: "
                        f"{scene}/{output_variant}"
                    )
                if source_variant == "identity":
                    with Image.open(source_path) as opened:
                        source_pixels = np.asarray(opened.convert("RGB"))
                    if not np.array_equal(repaired_pixels, source_pixels):
                        raise ValueError(
                            f"canonical identity pixels changed: {scene}/{expected_name}"
                        )
                    if not np.all(mask_pixels == 255):
                        raise ValueError(
                            f"canonical identity mask is not fully valid: "
                            f"{scene}/{expected_name}"
                        )
                    identity_pixel_matches += 1
                for path in (output_path_full, mask_path_full):
                    relative = path.relative_to(repaired_root).as_posix()
                    tree_digest.update(relative.encode("utf-8") + b"\0")
                    tree_digest.update(bytes.fromhex(_sha256(path)))
                image_count += 1
                mask_count += 1
        actual_files = {
            path.relative_to(repaired_root / scene)
            for path in (repaired_root / scene).rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError(
                f"repair file design mismatch for {scene}: "
                f"missing={sorted(str(path) for path in expected_files - actual_files)}, "
                f"extra={sorted(str(path) for path in actual_files - expected_files)}"
            )

    report = {
        "schema_version": "1.0",
        "status": "complete",
        "prepared_root": str(prepared_root.resolve()),
        "repaired_root": str(repaired_root.resolve()),
        "scene_count": len(scene_names),
        "scenes": scene_names,
        "source_variants": variant_names,
        "fill_policy": fill_policy,
        "output_variants": [
            _output_variant(variant, prefix) for variant in variant_names
        ],
        "image_count": image_count,
        "mask_count": mask_count,
        "identity_pixel_matches": identity_pixel_matches,
        "valid_fraction_by_variant": {
            variant: {
                "count": len(values),
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            }
            for variant, values in valid_fractions.items()
        },
        "tree_sha256": tree_digest.hexdigest(),
    }
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
