"""Scene-level preprocessing variants and reproducible manifests."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .image_ops import apply_affine
from .transforms import ImageAffine, crop_resize, letterbox, resize

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class TransformRecord:
    source: str
    output: str
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    matrix: list[list[float]]


@dataclass(frozen=True)
class VariantRecord:
    name: str
    seed: int
    interpolation: str
    images: list[TransformRecord]


def list_images(scene_dir: Path, max_views: int | None = None) -> list[Path]:
    images = sorted(
        path
        for path in scene_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if max_views is not None:
        images = images[:max_views]
    if len(images) < 2:
        raise ValueError("a scene must contain at least two supported images")
    return images


def protocol_affines(
    scene_dir: Path, image_paths: list[Path]
) -> list[np.ndarray]:
    """Read the exact source-to-prepared transform for selected inputs.

    Hand-authored image folders are treated as their own source domain. A
    protocol-generated variant must have a unique manifest record for every
    selected image; silently substituting identity for a partial manifest
    would invalidate the common-coordinate comparison.
    """

    manifest_path = scene_dir.parent / "manifest.json"
    if not manifest_path.exists():
        return [np.eye(3, dtype=np.float64) for _ in image_paths]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = [
        item for item in manifest["variants"] if item["name"] == scene_dir.name
    ]
    if len(variants) != 1:
        raise RuntimeError(f"manifest has no unique variant named {scene_dir.name}")
    records = {Path(item["output"]).name: item for item in variants[0]["images"]}
    missing = [path.name for path in image_paths if path.name not in records]
    if missing:
        raise RuntimeError(f"manifest is missing prepared inputs: {missing}")
    return [
        np.asarray(records[path.name]["matrix"], dtype=np.float64)
        for path in image_paths
    ]


def _center_crop(source_size: tuple[int, int], fraction: float) -> ImageAffine:
    width, height = source_size
    crop_width = width * fraction
    crop_height = height * fraction
    return crop_resize(
        source_size,
        (
            (width - crop_width) / 2.0,
            (height - crop_height) / 2.0,
            crop_width,
            crop_height,
        ),
        source_size,
    )


def _asymmetric_crop(
    source_size: tuple[int, int], fraction: float, rng: random.Random
) -> ImageAffine:
    width, height = source_size
    crop_width = width * fraction
    crop_height = height * fraction
    max_x = width - crop_width
    max_y = height - crop_height
    x = rng.uniform(0.0, max_x)
    y = rng.uniform(0.0, max_y)
    return crop_resize(source_size, (x, y, crop_width, crop_height), source_size)


def build_transform(
    variant: str,
    source_size: tuple[int, int],
    *,
    rng: random.Random,
) -> ImageAffine:
    if variant == "identity":
        return resize(source_size, source_size)
    if variant.startswith("center_crop_"):
        return _center_crop(source_size, _fraction_suffix(variant, "center_crop_"))
    if variant.startswith("asymmetric_crop_"):
        return _asymmetric_crop(
            source_size, _fraction_suffix(variant, "asymmetric_crop_"), rng
        )
    if variant == "letterbox_square":
        side = max(source_size)
        return letterbox(source_size, (side, side))
    raise ValueError(f"unknown protocol variant: {variant}")


def _fraction_suffix(variant: str, prefix: str) -> float:
    encoded = variant.removeprefix(prefix)
    if not encoded.isdigit() or len(encoded) != 3:
        raise ValueError(
            f"crop variant must encode a three-digit percentage, got {variant}"
        )
    fraction = int(encoded) / 100.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"crop fraction must be in (0, 1], got {fraction}")
    return fraction


def prepare_scene(
    scene_dir: Path,
    output_dir: Path,
    *,
    variants: Iterable[str],
    seed: int,
    max_views: int | None = None,
) -> dict[str, object]:
    sources = list_images(scene_dir, max_views=max_views)
    variant_records: list[VariantRecord] = []

    for variant_index, variant in enumerate(variants):
        rng = random.Random(seed + 10_007 * variant_index)
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        image_records: list[TransformRecord] = []

        for source in sources:
            with Image.open(source) as opened:
                image = opened.convert("RGB")
            transform = build_transform(variant, image.size, rng=rng)
            rendered = apply_affine(image, transform)
            output_name = f"{source.stem}.png"
            output_path = variant_dir / output_name
            rendered.save(output_path, format="PNG", optimize=False)
            image_records.append(
                TransformRecord(
                    source=source.name,
                    output=str(output_path.relative_to(output_dir)),
                    source_size=transform.source_size,
                    target_size=transform.target_size,
                    matrix=transform.matrix.tolist(),
                )
            )

        variant_records.append(
            VariantRecord(
                name=variant,
                seed=seed + 10_007 * variant_index,
                interpolation="Pillow bicubic",
                images=image_records,
            )
        )

    manifest = {
        "protocol_version": "0.1.0",
        "scene": scene_dir.name,
        "seed": seed,
        "source_directory": str(scene_dir.resolve()),
        "variants": [asdict(record) for record in variant_records],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2) + "\n"
    normalized_manifest = json.loads(rendered)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(rendered, encoding="utf-8")
    return normalized_manifest
