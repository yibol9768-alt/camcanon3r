#!/usr/bin/env python3
"""Summarize complete-design prediction timing and peak-memory metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--require-end-to-end", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "total": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("compute metadata must be finite and non-negative")
    return {
        "count": len(array),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "total": float(np.sum(array)),
    }


def _model_load_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("model-load metadata must be finite and non-negative")
    return {
        "record_count": len(array),
        "distinct_recorded_values": len(set(array.tolist())),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def summarize_prediction_compute(
    prediction_root: Path,
    *,
    model: str,
    dataset: str,
    scenes: list[str],
    variants: list[str],
    require_end_to_end: bool = False,
) -> dict[str, object]:
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("compute summary scenes must be non-empty and unique")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("compute summary variants must be non-empty and unique")
    expected = {
        (prediction_root / scene / f"{variant}.json").resolve()
        for scene in scenes
        for variant in variants
    }
    actual = {path.resolve() for path in prediction_root.rglob("*.json")}
    if actual != expected:
        raise ValueError(
            "prediction metadata design mismatch: "
            f"missing={sorted(str(path) for path in expected - actual)}, "
            f"extra={sorted(str(path) for path in actual - expected)}"
        )

    model_compute: list[float] = []
    end_to_end: list[float] = []
    model_load: list[float] = []
    peak_vram: list[int] = []
    records: list[dict[str, object]] = []
    for scene in scenes:
        for variant in variants:
            path = prediction_root / scene / f"{variant}.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            inputs = metadata.get("inputs")
            if not isinstance(inputs, list) or len(inputs) < 2:
                raise ValueError(f"prediction metadata has invalid inputs: {path}")
            core = metadata.get("model_compute_seconds")
            if core is None:
                if "inference_seconds" in metadata:
                    core = metadata["inference_seconds"]
                elif {
                    "pairwise_inference_seconds",
                    "alignment_seconds",
                }.issubset(metadata):
                    core = float(metadata["pairwise_inference_seconds"]) + float(
                        metadata["alignment_seconds"]
                    )
                else:
                    raise ValueError(f"prediction has no model-compute timing: {path}")
            core_value = float(core)
            load_raw = metadata.get("model_load_seconds")
            if load_raw is None:
                load_raw = metadata["load_seconds"]
            load_value = float(load_raw)
            end_value_raw = metadata.get(
                "end_to_end_seconds_excluding_model_load_and_metadata_write"
            )
            end_value = float(end_value_raw) if end_value_raw is not None else None
            peak_value = int(metadata["peak_vram_bytes"])
            if peak_value < 0:
                raise ValueError(f"prediction peak VRAM is negative: {path}")
            model_compute.append(core_value)
            model_load.append(load_value)
            peak_vram.append(peak_value)
            if end_value is not None:
                end_to_end.append(end_value)
            records.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "metadata": str(path.resolve()),
                    "metadata_sha256": _sha256(path),
                    "prediction_schema_version": metadata.get(
                        "prediction_schema_version"
                    ),
                    "view_count": len(inputs),
                    "model_compute_seconds": core_value,
                    "end_to_end_seconds_excluding_model_load_and_metadata_write": (
                        end_value
                    ),
                    "model_load_seconds": load_value,
                    "peak_vram_bytes": peak_value,
                }
            )
    expected_count = len(scenes) * len(variants)
    if len(records) != expected_count:
        raise RuntimeError("compute summary did not consume the complete design")
    if require_end_to_end and len(end_to_end) != expected_count:
        raise ValueError(
            "complete end-to-end timing is required: "
            f"expected={expected_count}, actual={len(end_to_end)}"
        )
    return {
        "schema_version": "1.0",
        "status": "complete",
        "model": model,
        "dataset": dataset,
        "prediction_root": str(prediction_root.resolve()),
        "scene_count": len(scenes),
        "variant_count": len(variants),
        "prediction_count": expected_count,
        "scenes": scenes,
        "variants": variants,
        "timing_boundaries": {
            "model_compute_seconds": (
                "VGGT forward inference or DUSt3R pairwise inference plus global alignment"
            ),
            "end_to_end_seconds_excluding_model_load_and_metadata_write": (
                "per-scene input hashing, image IO, preprocessing, model compute, "
                "postprocessing, and compressed NPZ write; excludes one-time model "
                "load and final metadata JSON write"
            ),
            "model_load_seconds": "one-time batch model construction and weight load",
        },
        "model_compute_seconds": _summary(model_compute),
        "end_to_end_seconds_excluding_model_load_and_metadata_write": _summary(
            end_to_end
        ),
        "end_to_end_available_count": len(end_to_end),
        "end_to_end_required": require_end_to_end,
        "model_load_seconds": _model_load_summary(model_load),
        "peak_vram_bytes": {
            "count": len(peak_vram),
            "median": float(np.median(peak_vram)),
            "maximum": max(peak_vram),
        },
        "records": records,
    }


def main() -> None:
    args = parse_args()
    report = summarize_prediction_compute(
        args.prediction_root,
        model=args.model,
        dataset=args.dataset,
        scenes=args.scenes,
        variants=args.variants,
        require_end_to_end=args.require_end_to_end,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "prediction_count": report["prediction_count"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
