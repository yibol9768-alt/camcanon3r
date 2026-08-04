"""Audit deterministic prediction arrays for an exact-input clean control."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .prediction import write_json_atomic

_PROTOCOL_METADATA_FIELDS = (
    "inputs",
    "spatial_transforms",
    "seed",
    "preprocess",
    "image_size",
    "batch_size",
    "schedule",
    "lr",
    "alignment",
    "scene_graph",
    "model",
    "weights",
    "input_tensor_shape",
    "dtype",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype.kind in "fc" and second.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def audit_prediction_repeat(
    reference_root: Path,
    candidate_root: Path,
    *,
    scenes: Sequence[str],
    variant: str = "identity",
    output_path: Path | None = None,
) -> dict[str, object]:
    scene_names = [str(scene) for scene in scenes]
    if not scene_names or len(set(scene_names)) != len(scene_names):
        raise ValueError("prediction repeat scenes must be non-empty and unique")
    expected_scenes = set(scene_names)
    for label, root in (("reference", reference_root), ("candidate", candidate_root)):
        actual_scenes = {path.name for path in root.iterdir() if path.is_dir()}
        if actual_scenes != expected_scenes:
            raise ValueError(
                f"{label} prediction scene design mismatch: "
                f"missing={sorted(expected_scenes - actual_scenes)}, "
                f"extra={sorted(actual_scenes - expected_scenes)}"
            )

    records: list[dict[str, object]] = []
    exact_scene_count = 0
    exact_array_count = 0
    array_count = 0
    for scene in scene_names:
        reference = reference_root / scene / f"{variant}.npz"
        candidate = candidate_root / scene / f"{variant}.npz"
        for path in (reference, candidate):
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise FileNotFoundError(
                    f"complete repeat prediction is missing: {path}"
                )
        reference_metadata = json.loads(
            reference.with_suffix(".json").read_text(encoding="utf-8")
        )
        candidate_metadata = json.loads(
            candidate.with_suffix(".json").read_text(encoding="utf-8")
        )
        compared_metadata = {
            field: reference_metadata.get(field)
            for field in _PROTOCOL_METADATA_FIELDS
            if field in reference_metadata or field in candidate_metadata
        }
        actual_metadata = {
            field: candidate_metadata.get(field) for field in compared_metadata
        }
        if actual_metadata != compared_metadata:
            raise ValueError(
                f"prediction repeat protocol metadata changed for {scene}: "
                f"expected={compared_metadata}, actual={actual_metadata}"
            )
        field_records: list[dict[str, object]] = []
        with np.load(reference) as reference_data, np.load(candidate) as candidate_data:
            if set(reference_data.files) != set(candidate_data.files):
                raise ValueError(
                    f"prediction repeat array schema changed for {scene}: "
                    f"reference={sorted(reference_data.files)}, "
                    f"candidate={sorted(candidate_data.files)}"
                )
            scene_exact = True
            for field in sorted(reference_data.files):
                first = np.asarray(reference_data[field])
                second = np.asarray(candidate_data[field])
                if first.shape != second.shape or first.dtype != second.dtype:
                    raise ValueError(
                        f"prediction repeat array shape/dtype changed: {scene}/{field}"
                    )
                exact = _equal(first, second)
                maximum_absolute_difference: float | None = None
                if not exact and first.dtype.kind in "biufc":
                    finite = np.isfinite(first) & np.isfinite(second)
                    if np.any(finite):
                        maximum_absolute_difference = float(
                            np.max(
                                np.abs(
                                    first[finite].astype(np.complex128)
                                    - second[finite].astype(np.complex128)
                                )
                            )
                        )
                field_records.append(
                    {
                        "field": field,
                        "shape": list(first.shape),
                        "dtype": str(first.dtype),
                        "exact": exact,
                        "maximum_absolute_difference": maximum_absolute_difference,
                    }
                )
                array_count += 1
                exact_array_count += int(exact)
                scene_exact = scene_exact and exact
        exact_scene_count += int(scene_exact)
        records.append(
            {
                "scene": scene,
                "variant": variant,
                "reference": str(reference.resolve()),
                "candidate": str(candidate.resolve()),
                "reference_sha256": _sha256(reference),
                "candidate_sha256": _sha256(candidate),
                "protocol_metadata_fields": sorted(compared_metadata),
                "all_arrays_exact": scene_exact,
                "arrays": field_records,
            }
        )
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "reference_root": str(reference_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "variant": variant,
        "scene_count": len(scene_names),
        "exact_scene_count": exact_scene_count,
        "array_count": array_count,
        "exact_array_count": exact_array_count,
        "all_prediction_arrays_exact": exact_array_count == array_count,
        "records": records,
    }
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
