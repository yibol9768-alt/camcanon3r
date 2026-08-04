#!/usr/bin/env python3
"""Normalize frozen ETH3D/DTU main results into one paper-table source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VARIANTS = (
    "identity",
    "center_crop_075",
    "asymmetric_crop_075",
    "letterbox_square",
)
COMMON = {
    "rotation_degrees": "rotation_median_degrees",
    "translation_degrees": "translation_median_degrees",
    "focal_relative": "focal_relative_error_median",
    "principal_point_normalized": "principal_point_normalized_error_median",
}
ETH3D_DELTAS = {
    "rotation_degrees": "rotation_delta_from_identity_degrees",
    "translation_degrees": "translation_delta_from_identity_degrees",
    "focal_relative": "focal_relative_error_delta_from_identity",
    "principal_point_normalized": (
        "principal_point_normalized_error_delta_from_identity"
    ),
    "depth_abs_rel": "depth_abs_rel_delta_from_identity",
    "point_accuracy_millimeters": "point_accuracy_delta_from_identity_meters",
    "point_completeness_millimeters": ("point_completeness_delta_from_identity_meters"),
}
DTU_DELTAS = {
    "rotation_degrees": "rotation_median_degrees_delta_from_identity",
    "translation_degrees": "translation_median_degrees_delta_from_identity",
    "focal_relative": "focal_relative_error_median_delta_from_identity",
    "principal_point_normalized": (
        "principal_point_normalized_error_median_delta_from_identity"
    ),
    "point_accuracy_millimeters": (
        "point_accuracy_mean_millimeters_delta_from_identity"
    ),
    "point_completeness_millimeters": (
        "point_completeness_mean_millimeters_delta_from_identity"
    ),
}
SCALES = {
    "rotation_degrees": 1.0,
    "translation_degrees": 1.0,
    "focal_relative": 100.0,
    "principal_point_normalized": 100.0,
    "depth_abs_rel": 100.0,
    "point_accuracy_millimeters": 1.0,
    "point_completeness_millimeters": 1.0,
}
UNITS = {
    "rotation_degrees": "degrees",
    "translation_degrees": "degrees",
    "focal_relative": "percentage_points",
    "principal_point_normalized": "percent_image_diagonal",
    "depth_abs_rel": "percentage_points",
    "point_accuracy_millimeters": "millimeters",
    "point_completeness_millimeters": "millimeters",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--summary",
        action="append",
        nargs=3,
        metavar=("MODEL", "DATASET", "PATH"),
        required=True,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scaled_interval(
    metrics: dict[str, object], field: str, scale: float
) -> dict[str, float] | None:
    interval = metrics.get(field)
    if interval is None:
        return None
    return {
        key: float(interval[key]) * scale
        for key in ("point_estimate", "lower", "upper")
    }


def _schema(summary: dict[str, object]) -> str:
    if "depth_evaluated" in summary:
        return "eth3d"
    if "point_metric_protocol" in summary:
        return "dtu"
    raise ValueError(
        "unsupported evaluation summary schema: "
        f"protocol={summary.get('evaluation_protocol_version')!r}"
    )


def summarize_one(model: str, dataset: str, path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    schema = _schema(summary)
    by_variant = summary["by_variant"]
    missing = set(VARIANTS) - set(by_variant)
    if missing:
        raise ValueError(f"summary lacks confirmatory variants: {sorted(missing)}")
    deltas = ETH3D_DELTAS if schema == "eth3d" else DTU_DELTAS
    point_scale = 1000.0 if schema == "eth3d" else 1.0
    records: list[dict[str, object]] = []
    for variant in VARIANTS:
        variant_record = by_variant[variant]
        metrics = variant_record["scene_bootstrap"]["metrics"]
        absolute = {
            label: _scaled_interval(metrics, field, SCALES[label])
            for label, field in COMMON.items()
        }
        delta = {
            label: (
                _scaled_interval(
                    metrics,
                    deltas[label],
                    (point_scale if label.startswith("point_") else SCALES[label]),
                )
                if label in deltas
                else None
            )
            for label in SCALES
        }
        if variant == "identity":
            nonzero = {
                label: interval["point_estimate"]
                for label, interval in delta.items()
                if interval is not None and abs(interval["point_estimate"]) > 1e-12
            }
            if nonzero:
                raise ValueError(f"identity deltas are nonzero: {nonzero}")
        records.append(
            {
                "variant": variant,
                "scene_count": int(variant_record["scene_count"]),
                "absolute": absolute,
                "delta_from_identity": delta,
            }
        )
    return {
        "model": model,
        "dataset": dataset,
        "summary_schema": schema,
        "source": str(path),
        "source_sha256": _sha256(path),
        "scene_count": int(summary["scene_count"]),
        "variants": records,
    }


def main() -> None:
    args = parse_args()
    seen: set[tuple[str, str]] = set()
    summaries: list[dict[str, object]] = []
    for model, dataset, name in args.summary:
        identity = (model, dataset)
        if identity in seen:
            raise ValueError(f"duplicate model/dataset summary: {identity}")
        seen.add(identity)
        summaries.append(summarize_one(model, dataset, Path(name)))
    payload = {
        "schema_version": "1.0",
        "status": "complete",
        "models_are_separate": True,
        "variants": list(VARIANTS),
        "metric_units": UNITS,
        "summary_count": len(summaries),
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "summary_count": len(summaries),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
