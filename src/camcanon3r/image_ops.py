"""Deterministic image rendering for geometric preprocessing protocols."""

from __future__ import annotations

from PIL import Image

from .transforms import ImageAffine


def apply_affine(
    image: Image.Image,
    transform: ImageAffine,
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Render an ``ImageAffine`` with bicubic interpolation.

    Pillow expects an inverse map from output pixels to input pixels, whereas
    ``ImageAffine`` stores the forward source-to-target map.
    """

    if image.size != transform.source_size:
        raise ValueError(
            f"image size {image.size} does not match transform source size "
            f"{transform.source_size}"
        )
    inverse = transform.inverse.matrix
    coefficients = tuple(inverse[:2].reshape(-1).tolist())
    return image.transform(
        transform.target_size,
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )
