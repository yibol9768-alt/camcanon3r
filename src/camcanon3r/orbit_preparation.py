"""Byte-preserving canonical-placement orbit preparation and auditing."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .prediction import write_json_atomic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_orbit_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    orbit = protocol.get("orbit")
    if (
        protocol.get("method_name") != "canonical_orbit_projection"
        or not protocol.get("frozen_before_any_orbit_projection_ground_truth_result")
        or not isinstance(orbit, dict)
        or orbit.get("resample_visible_pixels") is not False
        or orbit.get("same_placement_for_all_views_within_a_run") is not True
    ):
        raise ValueError("orbit protocol is not the frozen support-preserving design")
    members = orbit.get("ordered_members")
    if not isinstance(members, list) or len(members) < 3:
        raise ValueError("orbit protocol requires at least three ordered members")
    labels = [str(member.get("label")) for member in members]
    if len(set(labels)) != len(labels):
        raise ValueError("orbit member labels must be unique")
    for member in members:
        placement = member.get("placement")
        if (
            not isinstance(placement, list)
            or len(placement) != 2
            or not all(isinstance(value, (int, float)) for value in placement)
            or not all(0.0 <= float(value) <= 1.0 for value in placement)
            or str(member.get("inverse_pair")) not in labels
        ):
            raise ValueError("orbit member placement or inverse pair is invalid")
    inverse = {str(member["label"]): str(member["inverse_pair"]) for member in members}
    if any(inverse.get(partner) != label for label, partner in inverse.items()):
        raise ValueError("orbit inverse-pair mapping must be an involution")
    canvas_scale = float(orbit.get("canvas_scale", 0.0))
    fill = orbit.get("fill_rgb")
    if canvas_scale <= 1.0:
        raise ValueError("orbit canvas scale must exceed one")
    if (
        not isinstance(fill, list)
        or len(fill) != 3
        or any(not isinstance(value, int) or not 0 <= value <= 255 for value in fill)
    ):
        raise ValueError("orbit fill must be three uint8-compatible integers")
    fusion = protocol.get("geometry_fusion")
    if (
        not isinstance(fusion, dict)
        or fusion.get("ground_truth_used") is not False
        or fusion.get("removed_crop_pixels_imputed") is not False
        or str(fusion.get("reference_member")) not in labels
        or not 2 <= int(fusion.get("minimum_members", 0)) <= len(labels)
        or int(fusion.get("tile_rows", 0)) <= 0
        or int(fusion.get("geometric_median_iterations", 0)) <= 0
        or float(fusion.get("maximum_confidence_ratio", 0.0)) <= 0.0
    ):
        raise ValueError("orbit geometry fusion protocol is invalid")
    geometry_promotion = protocol.get("geometry_promotion")
    if (
        not isinstance(geometry_promotion, dict)
        or not 0.0
        <= float(geometry_promotion.get("maximum_relative_degradation_per_metric", -1))
        < 1.0
        or not 0.0
        < float(
            geometry_promotion.get(
                "minimum_relative_improvement_for_at_least_one_geometry_metric",
                0,
            )
        )
        < 1.0
        or geometry_promotion.get("requires_camera_promotion_separately") is not True
    ):
        raise ValueError("orbit geometry promotion protocol is invalid")
    return protocol


def _unique_variant(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise TypeError("canonical manifest variants are invalid")
    matches = [record for record in variants if record.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"canonical manifest has no unique variant named {name}")
    return dict(matches[0])


def _target_size(source_size: tuple[int, int], canvas_scale: float) -> tuple[int, int]:
    return tuple(
        max(value + 1, math.ceil(value * canvas_scale)) for value in source_size
    )


def _offset(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    placement: Sequence[float],
) -> tuple[int, int]:
    free_x = target_size[0] - source_size[0]
    free_y = target_size[1] - source_size[1]
    return (
        math.floor(free_x * float(placement[0]) + 0.5),
        math.floor(free_y * float(placement[1]) + 0.5),
    )


def _paste_exact(
    source: Image.Image,
    *,
    target_size: tuple[int, int],
    offset: tuple[int, int],
    fill: tuple[int, ...] | int,
    mode: str,
) -> Image.Image:
    if source.mode != mode:
        raise ValueError(f"orbit source must have mode {mode}, got {source.mode}")
    canvas = Image.new(mode, target_size, color=fill)
    canvas.paste(source, offset)
    return canvas


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG", optimize=False)
    temporary.replace(path)


def prepare_canonical_orbit_scene(
    canonical_scene_root: Path,
    output_scene_root: Path,
    protocol_path: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Embed one canonical repaired scene at every frozen orbit placement."""

    protocol = load_orbit_protocol(protocol_path)
    orbit = protocol["orbit"]
    source_variant = str(orbit["source_variant"])
    source_manifest_path = canonical_scene_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    canonical_variant = _unique_variant(source_manifest, source_variant)
    records = canonical_variant.get("images")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("canonical orbit source requires at least two images")

    output_variants: list[dict[str, Any]] = []
    fill = tuple(int(value) for value in orbit["fill_rgb"])
    canvas_scale = float(orbit["canvas_scale"])
    for member in orbit["ordered_members"]:
        label = str(member["label"])
        variant_name = f"orbit_{label}"
        output_records = []
        for source_record in records:
            source_path = canonical_scene_root / str(source_record["output"])
            repair = source_record.get("repair")
            if (
                not isinstance(repair, dict)
                or repair.get("fill_policy") != "neutral_gray"
            ):
                raise ValueError(
                    "orbit input must be the neutral-gray canonical repair"
                )
            mask_path = canonical_scene_root / str(repair["valid_mask"])
            with Image.open(source_path) as opened:
                source = opened.convert("RGB")
            with Image.open(mask_path) as opened:
                mask = opened.copy()
            if mask.mode != "L" or mask.size != source.size:
                raise ValueError("canonical validity mask does not match its image")
            source_size = source.size
            target_size = _target_size(source_size, canvas_scale)
            offset = _offset(source_size, target_size, member["placement"])
            output_name = Path(str(source_record["output"])).name
            output_path = output_scene_root / variant_name / output_name
            output_mask_path = output_scene_root / "_masks" / variant_name / output_name
            if not (resume and output_path.is_file() and output_mask_path.is_file()):
                rendered = _paste_exact(
                    source,
                    target_size=target_size,
                    offset=offset,
                    fill=fill,
                    mode="RGB",
                )
                rendered_mask = _paste_exact(
                    mask,
                    target_size=target_size,
                    offset=offset,
                    fill=0,
                    mode="L",
                )
                _save_png_atomic(rendered, output_path)
                _save_png_atomic(rendered_mask, output_mask_path)
            matrix = [
                [1.0, 0.0, float(offset[0])],
                [0.0, 1.0, float(offset[1])],
                [0.0, 0.0, 1.0],
            ]
            output_records.append(
                {
                    "source": source_record["source"],
                    "output": str(output_path.relative_to(output_scene_root)),
                    "source_size": list(source_size),
                    "target_size": list(target_size),
                    "matrix": matrix,
                    "orbit": {
                        "source_variant": source_variant,
                        "source_image": str(source_path.resolve()),
                        "source_mask": str(mask_path.resolve()),
                        "member": label,
                        "inverse_pair": str(member["inverse_pair"]),
                        "placement": [float(value) for value in member["placement"]],
                        "offset_xy": list(offset),
                        "fill_rgb": list(fill),
                        "resampled": False,
                        "valid_mask": str(
                            output_mask_path.relative_to(output_scene_root)
                        ),
                    },
                }
            )
        output_variants.append(
            {
                "name": variant_name,
                "interpolation": "none; integer-coordinate byte-preserving paste",
                "images": output_records,
            }
        )

    manifest = {
        "protocol_version": "canonical-orbit-0.1",
        "scene": source_manifest.get("scene", canonical_scene_root.name),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "orbit_protocol": str(protocol_path.resolve()),
        "orbit_protocol_sha256": _sha256(protocol_path),
        "variants": output_variants,
    }
    output_scene_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_scene_root / "manifest.json", manifest)
    return manifest


def _outside_rectangle(
    target_size: tuple[int, int],
    source_size: tuple[int, int],
    offset: tuple[int, int],
) -> np.ndarray:
    outside = np.ones((target_size[1], target_size[0]), dtype=bool)
    x, y = offset
    outside[y : y + source_size[1], x : x + source_size[0]] = False
    return outside


def audit_canonical_orbit_scene(
    canonical_scene_root: Path,
    output_scene_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    """Verify exact support preservation and transform provenance for one scene."""

    protocol = load_orbit_protocol(protocol_path)
    orbit = protocol["orbit"]
    source_manifest_path = canonical_scene_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_variant = _unique_variant(source_manifest, str(orbit["source_variant"]))
    source_records = source_variant["images"]
    output_manifest_path = output_scene_root / "manifest.json"
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    expected_header = {
        "protocol_version": "canonical-orbit-0.1",
        "scene": source_manifest.get("scene", canonical_scene_root.name),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "orbit_protocol": str(protocol_path.resolve()),
        "orbit_protocol_sha256": _sha256(protocol_path),
    }
    if {key: output_manifest.get(key) for key in expected_header} != expected_header:
        raise ValueError("orbit manifest header does not match its frozen sources")
    variants = output_manifest.get("variants")
    expected_names = [f"orbit_{member['label']}" for member in orbit["ordered_members"]]
    if (
        not isinstance(variants, list)
        or [item.get("name") for item in variants] != expected_names
    ):
        raise ValueError("orbit manifest members do not match frozen order")

    fill = np.asarray(orbit["fill_rgb"], dtype=np.uint8)
    canvas_scale = float(orbit["canvas_scale"])
    tree = hashlib.sha256()
    image_count = 0
    mask_count = 0
    decoded_rgb_matches = 0
    decoded_mask_matches = 0
    for member, variant in zip(orbit["ordered_members"], variants, strict=True):
        output_records = variant.get("images")
        if not isinstance(output_records, list) or len(output_records) != len(
            source_records
        ):
            raise ValueError("orbit member image count is incomplete")
        for source_record, output_record in zip(
            source_records, output_records, strict=True
        ):
            source_path = canonical_scene_root / str(source_record["output"])
            repair = source_record["repair"]
            source_mask_path = canonical_scene_root / str(repair["valid_mask"])
            output_path = output_scene_root / str(output_record["output"])
            orbit_record = output_record.get("orbit")
            if not isinstance(orbit_record, dict):
                raise TypeError("orbit image provenance is missing")
            output_mask_path = output_scene_root / str(orbit_record["valid_mask"])
            with Image.open(source_path) as opened:
                source = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            with Image.open(source_mask_path) as opened:
                source_mask = np.asarray(opened, dtype=np.uint8)
            with Image.open(output_path) as opened:
                output = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            with Image.open(output_mask_path) as opened:
                output_mask = np.asarray(opened, dtype=np.uint8)
            source_size = (source.shape[1], source.shape[0])
            target_size = _target_size(source_size, canvas_scale)
            offset = _offset(source_size, target_size, member["placement"])
            x, y = offset
            visible = output[y : y + source_size[1], x : x + source_size[0]]
            visible_mask = output_mask[y : y + source_size[1], x : x + source_size[0]]
            if not np.array_equal(visible, source):
                raise ValueError("orbit member changed a decoded source RGB byte")
            if not np.array_equal(visible_mask, source_mask):
                raise ValueError("orbit member changed a decoded source mask byte")
            outside = _outside_rectangle(target_size, source_size, offset)
            if output.shape != (target_size[1], target_size[0], 3):
                raise ValueError("orbit image has the wrong canvas size")
            if output_mask.shape != (target_size[1], target_size[0]):
                raise ValueError("orbit mask has the wrong canvas size")
            if not np.all(output[outside] == fill):
                raise ValueError("orbit image has a non-frozen outer fill pixel")
            if np.any(output_mask[outside] != 0):
                raise ValueError("orbit mask marks an outer fill pixel as valid")
            expected_matrix = np.asarray(
                [[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]], [0.0, 0.0, 1.0]]
            )
            if (
                output_record.get("source") != source_record.get("source")
                or output_record.get("source_size") != list(source_size)
                or output_record.get("target_size") != list(target_size)
                or not np.array_equal(
                    np.asarray(output_record.get("matrix")), expected_matrix
                )
                or orbit_record.get("member") != member["label"]
                or orbit_record.get("inverse_pair") != member["inverse_pair"]
                or orbit_record.get("offset_xy") != list(offset)
                or orbit_record.get("resampled") is not False
            ):
                raise ValueError("orbit image record drifted from its frozen transform")
            for path in (output_path, output_mask_path):
                relative = path.relative_to(output_scene_root).as_posix()
                tree.update(relative.encode("utf-8") + b"\0")
                tree.update(bytes.fromhex(_sha256(path)))
            image_count += 1
            mask_count += 1
            decoded_rgb_matches += 1
            decoded_mask_matches += 1
    return {
        "schema_version": "canonical-orbit-audit-0.1",
        "status": "complete",
        "scene": expected_header["scene"],
        "member_count": len(variants),
        "source_image_count": len(source_records),
        "image_count": image_count,
        "mask_count": mask_count,
        "decoded_rgb_matches": decoded_rgb_matches,
        "decoded_mask_matches": decoded_mask_matches,
        "tree_sha256": tree.hexdigest(),
        "orbit_protocol_sha256": expected_header["orbit_protocol_sha256"],
        "source_manifest_sha256": expected_header["source_manifest_sha256"],
    }
