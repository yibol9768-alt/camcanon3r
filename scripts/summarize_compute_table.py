#!/usr/bin/env python3
"""Normalize frozen inference and repair compute into one paper-table source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--sweep",
        action="append",
        nargs=4,
        metavar=("MODEL", "DATASET", "ROLE", "PATH"),
        required=True,
    )
    parser.add_argument("--repair-ablation", type=Path)
    parser.add_argument("--canonicalization", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"compute field must be finite and non-negative: {label}")
    return result


def normalize_sweep(
    model: str, dataset: str, role: str, path: Path
) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete"
        or report.get("model") != model
        or report.get("dataset") != dataset
    ):
        raise ValueError(f"compute sweep provenance mismatch: {path}")
    prediction_count = int(report["prediction_count"])
    scene_count = int(report["scene_count"])
    variant_count = int(report["variant_count"])
    if prediction_count != scene_count * variant_count or prediction_count <= 0:
        raise ValueError(f"compute sweep design is incomplete: {path}")
    records = report["records"]
    if not isinstance(records, list) or len(records) != prediction_count:
        raise ValueError(f"compute sweep records are incomplete: {path}")
    view_counts = sorted({int(record["view_count"]) for record in records})
    if not view_counts or view_counts[0] < 2:
        raise ValueError(f"compute sweep view counts are invalid: {path}")
    model_compute = report["model_compute_seconds"]
    end_to_end = report["end_to_end_seconds_excluding_model_load_and_metadata_write"]
    peak_vram = report["peak_vram_bytes"]
    if int(model_compute["count"]) != prediction_count:
        raise ValueError(f"model compute does not cover the complete sweep: {path}")
    end_count = int(report["end_to_end_available_count"])
    if end_count not in (0, prediction_count) or int(end_to_end["count"]) != end_count:
        raise ValueError(f"end-to-end timing is partially available: {path}")
    if int(peak_vram["count"]) != prediction_count:
        raise ValueError(f"peak VRAM does not cover the complete sweep: {path}")
    load = report["model_load_seconds"]
    normalized = {
        "model": model,
        "dataset": dataset,
        "role": role,
        "scene_count": scene_count,
        "variant_count": variant_count,
        "prediction_count": prediction_count,
        "view_counts": view_counts,
        "model_compute_seconds": {
            key: _finite(model_compute[key], f"model_compute_seconds.{key}")
            for key in ("median", "p90", "total")
        },
        "end_to_end_seconds_excluding_model_load_and_metadata_write": (
            {
                key: _finite(end_to_end[key], f"end_to_end_seconds.{key}")
                for key in ("median", "p90", "total")
            }
            if end_count
            else None
        ),
        "end_to_end_availability": (
            "complete" if end_count == prediction_count else "legacy_unavailable"
        ),
        "model_load_seconds": {
            key: _finite(load[key], f"model_load_seconds.{key}")
            for key in ("median", "minimum", "maximum")
        },
        "peak_vram_gibibytes": {
            "median": _finite(peak_vram["median"], "peak_vram.median") / 2**30,
            "maximum": _finite(peak_vram["maximum"], "peak_vram.maximum") / 2**30,
        },
        "source": str(path),
        "source_sha256": _sha256(path),
    }
    return normalized


def normalize_repair_ablation(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or not report.get("models_are_separate"):
        raise ValueError("repair ablation compute source is not complete and separated")
    models: list[dict[str, object]] = []
    for model_record in report["models"]:
        methods: list[dict[str, object]] = []
        for record in model_record["records"]:
            methods.append(
                {
                    "method": str(record["method"]),
                    "kind": str(record["kind"]),
                    "model_runs_per_scene": int(record["model_runs_per_scene"]),
                    "median_model_compute_seconds_per_scene": _finite(
                        record["median_model_compute_seconds_per_scene"],
                        "repair.median_model_compute_seconds_per_scene",
                    ),
                    "maximum_peak_vram_gibibytes": _finite(
                        record["maximum_peak_vram_bytes"],
                        "repair.maximum_peak_vram_bytes",
                    )
                    / 2**30,
                }
            )
        models.append(
            {
                "model": str(model_record["model"]),
                "dataset": str(model_record["dataset"]),
                "scene_count": int(model_record["scene_count"]),
                "methods": methods,
            }
        )
    if not models or len({record["model"] for record in models}) != len(models):
        raise ValueError("repair ablation has no unique model records")
    return {
        "source": str(path),
        "source_sha256": _sha256(path),
        "models": models,
    }


def normalize_canonicalization(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report.get("records")
    if (
        report.get("status") != "complete"
        or not isinstance(records, list)
        or int(report.get("record_count", -1)) != len(records)
        or not records
    ):
        raise ValueError("canonicalization compute source is incomplete")
    grouped: dict[str, list[float]] = {}
    seen: set[tuple[str, str]] = set()
    for record in records:
        identity = (str(record["scene"]), str(record["source_variant"]))
        if identity in seen:
            raise ValueError(f"duplicate canonicalization compute record: {identity}")
        seen.add(identity)
        elapsed = _finite(
            record["canonicalization_seconds"], "canonicalization_seconds"
        )
        grouped.setdefault(identity[1], []).append(elapsed)
    declared = report["canonicalization_seconds"]
    total = sum(value for values in grouped.values() for value in values)
    if int(declared["count"]) != len(records) or not math.isclose(
        _finite(declared["total"], "canonicalization.total"),
        total,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("canonicalization compute aggregate does not match records")
    return {
        "source": str(path),
        "source_sha256": _sha256(path),
        "fill_policy": str(report["fill_policy"]),
        "timing_boundary": str(report["timing_boundary"]),
        "record_count": len(records),
        "by_source_variant": {
            variant: {
                "count": len(values),
                "median_seconds": float(median(values)),
                "maximum_seconds": max(values),
                "total_seconds": sum(values),
            }
            for variant, values in grouped.items()
        },
    }


def summarize_compute_table(
    sweeps: list[tuple[str, str, str, Path]],
    *,
    repair_ablation: Path | None = None,
    canonicalization: Path | None = None,
) -> dict[str, object]:
    seen: set[tuple[str, str, str]] = set()
    normalized: list[dict[str, object]] = []
    for model, dataset, role, path in sweeps:
        identity = (model, dataset, role)
        if identity in seen:
            raise ValueError(f"duplicate compute sweep: {identity}")
        seen.add(identity)
        normalized.append(normalize_sweep(model, dataset, role, path))
    if not normalized:
        raise ValueError("at least one compute sweep is required")
    return {
        "schema_version": "1.0",
        "status": "complete",
        "models_are_separate": True,
        "timing_units": "seconds",
        "memory_units": "GiB (2^30 bytes)",
        "sweep_count": len(normalized),
        "sweeps": normalized,
        "repair_ablation": (
            normalize_repair_ablation(repair_ablation)
            if repair_ablation is not None
            else None
        ),
        "canonicalization": (
            normalize_canonicalization(canonicalization)
            if canonicalization is not None
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    payload = summarize_compute_table(
        [
            (model, dataset, role, Path(name))
            for model, dataset, role, name in args.sweep
        ],
        repair_ablation=args.repair_ablation,
        canonicalization=args.canonicalization,
    )
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
                "sweep_count": payload["sweep_count"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
