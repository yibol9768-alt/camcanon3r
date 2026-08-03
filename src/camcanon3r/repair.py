"""Transform-aware canonical-canvas repair for prepared variants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .image_ops import apply_affine
from .transforms import ImageAffine

FILL_POLICIES = {"neutral_gray", "black", "image_mean"}


def _fill_color(image: Image.Image, policy: str) -> tuple[int, int, int]:
    if policy not in FILL_POLICIES:
        raise ValueError(f"unknown canonical fill policy: {policy}")
    if policy == "black":
        return (0, 0, 0)
    if policy == "neutral_gray":
        return (127, 127, 127)
    pixels = np.asarray(image, dtype=np.float64)
    return tuple(round(value) for value in pixels.mean(axis=(0, 1)))


def _load_variant_record(variant_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = variant_dir.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"protocol manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        item for item in manifest["variants"] if item["name"] == variant_dir.name
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest has no unique variant named {variant_dir.name}")
    return manifest, matches[0]


def _merge_repair_manifest(
    output_root: Path,
    *,
    scene: str,
    source_manifest: Path,
    variant: dict[str, object],
) -> None:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("scene") != scene:
            raise ValueError("repair manifest scene does not match source scene")
        variants = [
            item for item in manifest["variants"] if item["name"] != variant["name"]
        ]
    else:
        manifest = {
            "protocol_version": "0.2.0",
            "scene": scene,
            "source_manifest": str(source_manifest.resolve()),
        }
        variants = []
    variants.append(variant)
    manifest["variants"] = variants
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def canonicalize_variant(
    variant_dir: Path,
    output_dir: Path,
    *,
    fill_policy: str = "neutral_gray",
) -> dict[str, object]:
    """Undo a known protocol affine onto the original camera canvas.

    The inverse warp restores the original pixel coordinate system but cannot
    recreate content outside a crop. Missing support is filled deterministically
    and exported as a binary mask. The repaired manifest therefore records an
    identity source-to-prepared affine, while preserving the original affine
    and valid fraction as repair provenance.
    """

    source_manifest, variant = _load_variant_record(variant_dir)
    if not variant["images"]:
        raise ValueError("source variant contains no images")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir.parent / "_masks" / output_dir.name
    mask_dir.mkdir(parents=True, exist_ok=True)
    rendered_records: list[dict[str, object]] = []

    for record in variant["images"]:
        input_path = variant_dir.parent / record["output"]
        if not input_path.is_file():
            raise FileNotFoundError(f"prepared input is missing: {input_path}")
        with Image.open(input_path) as opened:
            image = opened.convert("RGB")
        source_size = tuple(int(value) for value in record["source_size"])
        prepared_size = tuple(int(value) for value in record["target_size"])
        if image.size != prepared_size:
            raise ValueError(
                f"prepared image size {image.size} disagrees with manifest "
                f"size {prepared_size}"
            )
        source_to_prepared = ImageAffine(
            np.asarray(record["matrix"], dtype=np.float64),
            source_size,
            prepared_size,
        )
        prepared_to_source = source_to_prepared.inverse
        fill = _fill_color(image, fill_policy)
        repaired = apply_affine(image, prepared_to_source, fill=fill)
        valid_input = Image.new("L", prepared_size, color=255)
        valid = apply_affine(
            valid_input,
            prepared_to_source,
            fill=0,
            resample=Image.Resampling.NEAREST,
        )
        output_name = Path(record["output"]).name
        output_path = output_dir / output_name
        mask_path = mask_dir / output_name
        repaired.save(output_path, format="PNG", optimize=False)
        valid.save(mask_path, format="PNG", optimize=False)
        valid_fraction = float(np.asarray(valid, dtype=np.uint8).mean() / 255.0)
        rendered_records.append(
            {
                "source": record["source"],
                "output": str(output_path.relative_to(output_dir.parent)),
                "source_size": list(source_size),
                "target_size": list(source_size),
                "matrix": np.eye(3).tolist(),
                "repair": {
                    "method": "known_affine_inverse_canvas",
                    "input_variant": variant_dir.name,
                    "input_affine": source_to_prepared.matrix.tolist(),
                    "fill_policy": fill_policy,
                    "fill_rgb": list(fill),
                    "valid_mask": str(mask_path.relative_to(output_dir.parent)),
                    "valid_fraction": valid_fraction,
                },
            }
        )

    repaired_variant = {
        "name": output_dir.name,
        "seed": variant.get("seed"),
        "interpolation": "Pillow bicubic; nearest-neighbor validity mask",
        "images": rendered_records,
    }
    _merge_repair_manifest(
        output_dir.parent,
        scene=source_manifest["scene"],
        source_manifest=variant_dir.parent / "manifest.json",
        variant=repaired_variant,
    )
    return repaired_variant
