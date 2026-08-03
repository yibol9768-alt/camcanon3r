"""Exact spatial-size bookkeeping for the official VGGT image loader."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import ImageAffine


@dataclass(frozen=True)
class VggtPreprocessSpec:
    """One image's nominal source-to-batched-tensor pixel transform."""

    affine: ImageAffine
    resized_size: tuple[int, int]
    crop_top: int
    padding: tuple[int, int, int, int]


def plan_vggt_preprocessing(
    source_sizes: list[tuple[int, int]],
    *,
    mode: str,
    target_size: int = 518,
) -> list[VggtPreprocessSpec]:
    """Mirror ``vggt.utils.load_fn.load_and_preprocess_images`` geometry.

    The official loader rounds the aspect-preserving dimension to a multiple
    of 14, optionally center-crops, and finally pads unequal batch shapes. This
    function records that full nominal affine so downstream evaluation does
    not mistake hidden model preprocessing for reconstruction drift.
    """

    if not source_sizes:
        raise ValueError("at least one source size is required")
    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be 'crop' or 'pad'")
    if target_size <= 0:
        raise ValueError("target size must be positive")

    intermediate: list[dict[str, object]] = []
    for width, height in source_sizes:
        if width <= 0 or height <= 0:
            raise ValueError("source dimensions must be positive")
        if mode == "crop" or width >= height:
            resized_width = target_size
            resized_height = round(height * (resized_width / width) / 14) * 14
        else:
            resized_height = target_size
            resized_width = round(width * (resized_height / height) / 14) * 14
        if resized_width <= 0 or resized_height <= 0:
            raise ValueError("official VGGT rounding produced a zero-sized image")

        crop_top = (
            (resized_height - target_size) // 2
            if mode == "crop" and resized_height > target_size
            else 0
        )
        cropped_width = resized_width
        cropped_height = min(resized_height, target_size) if mode == "crop" else resized_height

        if mode == "pad":
            height_padding = target_size - cropped_height
            width_padding = target_size - cropped_width
            pad_top = height_padding // 2
            pad_bottom = height_padding - pad_top
            pad_left = width_padding // 2
            pad_right = width_padding - pad_left
        else:
            pad_top = pad_bottom = pad_left = pad_right = 0

        intermediate.append(
            {
                "source_size": (width, height),
                "resized_size": (resized_width, resized_height),
                "crop_top": crop_top,
                "shape": (
                    cropped_width + pad_left + pad_right,
                    cropped_height + pad_top + pad_bottom,
                ),
                "padding": (pad_left, pad_top, pad_right, pad_bottom),
            }
        )

    batch_width = max(int(item["shape"][0]) for item in intermediate)  # type: ignore[index]
    batch_height = max(int(item["shape"][1]) for item in intermediate)  # type: ignore[index]
    specs: list[VggtPreprocessSpec] = []
    for item in intermediate:
        width, height = item["source_size"]  # type: ignore[misc]
        resized_width, resized_height = item["resized_size"]  # type: ignore[misc]
        shape_width, shape_height = item["shape"]  # type: ignore[misc]
        pad_left, pad_top, pad_right, pad_bottom = item["padding"]  # type: ignore[misc]
        batch_pad_left = (batch_width - shape_width) // 2
        batch_pad_right = batch_width - shape_width - batch_pad_left
        batch_pad_top = (batch_height - shape_height) // 2
        batch_pad_bottom = batch_height - shape_height - batch_pad_top
        pad_left += batch_pad_left
        pad_right += batch_pad_right
        pad_top += batch_pad_top
        pad_bottom += batch_pad_bottom
        crop_top = int(item["crop_top"])
        matrix = np.array(
            [
                [resized_width / width, 0.0, pad_left],
                [0.0, resized_height / height, pad_top - crop_top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        specs.append(
            VggtPreprocessSpec(
                affine=ImageAffine(
                    matrix=matrix,
                    source_size=(width, height),
                    target_size=(batch_width, batch_height),
                ),
                resized_size=(resized_width, resized_height),
                crop_top=crop_top,
                padding=(pad_left, pad_top, pad_right, pad_bottom),
            )
        )
    return specs
