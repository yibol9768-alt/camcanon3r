"""Image-space affine transforms and their exact camera updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _matrix3(value: ArrayLike) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains a non-finite value")
    return matrix


@dataclass(frozen=True)
class ImageAffine:
    """A pixel-coordinate affine map from a source image to a target image.

    Pixel centers use homogeneous coordinates ``[x, y, 1]``. If a camera has
    intrinsics ``K`` before preprocessing, its exact target intrinsics are
    ``A @ K``.
    """

    matrix: FloatArray
    source_size: tuple[int, int]
    target_size: tuple[int, int]

    def __post_init__(self) -> None:
        matrix = _matrix3(self.matrix)
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError("image transform must be affine in pixel coordinates")
        for name, size in (("source", self.source_size), ("target", self.target_size)):
            if len(size) != 2 or min(size) <= 0:
                raise ValueError(
                    f"{name} size must be (width, height) with positive values"
                )
        object.__setattr__(self, "matrix", matrix)

    @property
    def inverse(self) -> ImageAffine:
        return ImageAffine(
            matrix=np.linalg.inv(self.matrix),
            source_size=self.target_size,
            target_size=self.source_size,
        )

    def transform_intrinsics(self, intrinsics: ArrayLike) -> FloatArray:
        return self.matrix @ _matrix3(intrinsics)

    def inverse_intrinsics(self, transformed_intrinsics: ArrayLike) -> FloatArray:
        return np.linalg.solve(self.matrix, _matrix3(transformed_intrinsics))

    def transform_pixels(self, pixels: ArrayLike) -> FloatArray:
        points = np.asarray(pixels, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError("pixels must have shape (..., 2)")
        homogeneous = np.concatenate(
            [points, np.ones((*points.shape[:-1], 1))], axis=-1
        )
        mapped = homogeneous @ self.matrix.T
        return mapped[..., :2] / mapped[..., 2:3]

    def compose(self, after: ImageAffine) -> ImageAffine:
        """Apply this transform and then ``after``."""

        if self.target_size != after.source_size:
            raise ValueError("transform sizes do not compose")
        return ImageAffine(
            matrix=after.matrix @ self.matrix,
            source_size=self.source_size,
            target_size=after.target_size,
        )


def resize(source_size: tuple[int, int], target_size: tuple[int, int]) -> ImageAffine:
    source_width, source_height = source_size
    target_width, target_height = target_size
    matrix = np.array(
        [
            [target_width / source_width, 0.0, 0.0],
            [0.0, target_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ImageAffine(matrix, source_size, target_size)


def crop_resize(
    source_size: tuple[int, int],
    crop_xywh: tuple[float, float, float, float],
    target_size: tuple[int, int],
) -> ImageAffine:
    """Crop ``(x, y, width, height)`` and resize it to ``target_size``."""

    source_width, source_height = source_size
    x, y, crop_width, crop_height = crop_xywh
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop dimensions must be positive")
    if (
        x < 0
        or y < 0
        or x + crop_width > source_width
        or y + crop_height > source_height
    ):
        raise ValueError("crop lies outside the source image")
    target_width, target_height = target_size
    scale_x = target_width / crop_width
    scale_y = target_height / crop_height
    matrix = np.array(
        [
            [scale_x, 0.0, -scale_x * x],
            [0.0, scale_y, -scale_y * y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ImageAffine(matrix, source_size, target_size)


def letterbox(
    source_size: tuple[int, int], target_size: tuple[int, int]
) -> ImageAffine:
    """Aspect-preserving resize followed by symmetric target-space padding."""

    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = min(target_width / source_width, target_height / source_height)
    pad_x = (target_width - scale * source_width) / 2.0
    pad_y = (target_height - scale * source_height) / 2.0
    matrix = np.array(
        [[scale, 0.0, pad_x], [0.0, scale, pad_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return ImageAffine(matrix, source_size, target_size)
