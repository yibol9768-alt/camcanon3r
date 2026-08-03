"""Exact pixel bookkeeping for the official DUSt3R image loader."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import ImageAffine


@dataclass(frozen=True)
class Dust3rPreprocessSpec:
    """One image's prepared-domain to DUSt3R tensor transform."""

    affine: ImageAffine
    resized_size: tuple[int, int]
    crop_left_top_right_bottom: tuple[int, int, int, int]


def plan_dust3r_preprocessing(
    source_sizes: list[tuple[int, int]],
    *,
    image_size: int = 512,
    patch_size: int = 16,
    square_ok: bool = False,
) -> list[Dust3rPreprocessSpec]:
    """Mirror ``dust3r.utils.image.load_images`` at pixel level.

    DUSt3R independently rounds the aspect-preserving resize dimensions and
    then center-crops them to patch multiples. Its default 512-pixel model also
    converts a square input to a 4:3 tensor unless ``square_ok`` is true. The
    resulting anisotropy from integer resize rounding is retained exactly.
    """

    if not source_sizes:
        raise ValueError("at least one source size is required")
    if image_size not in {224, 512}:
        raise ValueError("official DUSt3R image size must be 224 or 512")
    if patch_size <= 0:
        raise ValueError("patch size must be positive")

    specs: list[Dust3rPreprocessSpec] = []
    for width, height in source_sizes:
        if width <= 0 or height <= 0:
            raise ValueError("source dimensions must be positive")
        longest = max(width, height)
        if image_size == 224:
            resize_long_edge = round(image_size * max(width / height, height / width))
        else:
            resize_long_edge = image_size
        resized_width = round(width * resize_long_edge / longest)
        resized_height = round(height * resize_long_edge / longest)
        center_x = resized_width // 2
        center_y = resized_height // 2
        if image_size == 224:
            half_width = half_height = min(center_x, center_y)
        else:
            half_width = ((2 * center_x) // patch_size) * patch_size // 2
            half_height = ((2 * center_y) // patch_size) * patch_size // 2
            if not square_ok and resized_width == resized_height:
                half_height = 3 * half_width // 4
        left = center_x - half_width
        top = center_y - half_height
        right = center_x + half_width
        bottom = center_y + half_height
        target_size = (right - left, bottom - top)
        if min(target_size) <= 0:
            raise ValueError("official DUSt3R crop produced a zero-sized image")
        matrix = np.array(
            [
                [resized_width / width, 0.0, -left],
                [0.0, resized_height / height, -top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        specs.append(
            Dust3rPreprocessSpec(
                affine=ImageAffine(matrix, (width, height), target_size),
                resized_size=(resized_width, resized_height),
                crop_left_top_right_bottom=(left, top, right, bottom),
            )
        )
    return specs
