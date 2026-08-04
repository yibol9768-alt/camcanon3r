"""Integrity checks for the support-preserving letterbox control."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .prediction import write_json_atomic
from .prepared_audit import audit_prepared_sweep

SUPPORT_VARIANTS = (
    "letterbox_square",
    "shared_asymmetric_letterbox_square",
    "asymmetric_letterbox_square",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_support_control(
    prepared_root: Path,
    reference_root: Path,
    variant_config_path: Path,
    scene_images: Mapping[str, Sequence[str]],
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Audit content preservation and the symmetric-letterbox anchor."""

    config = json.loads(variant_config_path.read_text(encoding="utf-8"))
    if (
        not config.get("frozen_before_dtu_gt_inspection")
        or not config.get("frozen_before_support_control_results")
        or not config.get("registered_after_eth3d_mechanism_results")
        or config.get("frozen_before_benchmark_scale_mechanism_results") is not False
        or config.get("experiment_role") != "support_preserving_coordinate_control"
        or tuple(config.get("ordered_variants", ())) != SUPPORT_VARIANTS
        or config.get("anchor_variant") != "letterbox_square"
    ):
        raise ValueError("support-control variant design is not the frozen protocol")
    report = audit_prepared_sweep(
        prepared_root,
        variant_config_path,
        scene_images,
    )
    reference_matches = 0
    reference_digest = hashlib.sha256()
    for scene in sorted(scene_images):
        support_manifest_path = prepared_root / scene / "manifest.json"
        reference_manifest_path = reference_root / scene / "manifest.json"
        if not reference_manifest_path.is_file():
            raise FileNotFoundError(
                f"support-control reference manifest is missing: "
                f"{reference_manifest_path}"
            )
        support_manifest = json.loads(support_manifest_path.read_text(encoding="utf-8"))
        reference_manifest = json.loads(
            reference_manifest_path.read_text(encoding="utf-8")
        )
        support_anchor = [
            record
            for record in support_manifest["variants"]
            if record["name"] == "letterbox_square"
        ]
        reference_anchor = [
            record
            for record in reference_manifest["variants"]
            if record["name"] == "letterbox_square"
        ]
        if len(support_anchor) != 1 or len(reference_anchor) != 1:
            raise ValueError(f"letterbox anchor is not unique for scene {scene}")
        support_images = support_anchor[0]["images"]
        reference_images = reference_anchor[0]["images"]
        if len(support_images) != len(scene_images[scene]) or len(
            reference_images
        ) != len(scene_images[scene]):
            raise ValueError(f"letterbox anchor view count drift: {scene}")
        for support_record, reference_record, source_name in zip(
            support_images,
            reference_images,
            scene_images[scene],
            strict=True,
        ):
            comparable_fields = (
                "source",
                "output",
                "source_size",
                "target_size",
                "matrix",
            )
            if (
                any(
                    support_record[field] != reference_record[field]
                    for field in comparable_fields
                )
                or support_record["source"] != source_name
            ):
                raise ValueError(
                    f"letterbox anchor manifest drift: {scene}/{source_name}"
                )
            support_path = prepared_root / scene / str(support_record["output"])
            reference_path = reference_root / scene / str(reference_record["output"])
            support_hash = _sha256(support_path)
            if support_hash != _sha256(reference_path):
                raise ValueError(f"letterbox anchor image drift: {scene}/{source_name}")
            relative = support_path.relative_to(prepared_root).as_posix()
            reference_digest.update(relative.encode("utf-8") + b"\0")
            reference_digest.update(bytes.fromhex(support_hash))
            reference_matches += 1
    expected_matches = sum(len(images) for images in scene_images.values())
    if reference_matches != expected_matches:
        raise RuntimeError("support-control anchor audit is incomplete")
    report.update(
        {
            "schema_version": "support-control-1.0",
            "variant_config": str(variant_config_path.resolve()),
            "variant_config_sha256": _sha256(variant_config_path),
            "reference_root": str(reference_root.resolve()),
            "reference_letterbox_matches": reference_matches,
            "reference_letterbox_tree_sha256": reference_digest.hexdigest(),
        }
    )
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
