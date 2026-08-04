#!/usr/bin/env python3
"""Build one provenance-bound repair ablation table from frozen reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROTATION = "relative_rotation_median_degrees"
DEPTH = "depth_mean_abs_rel"
FILL_LABELS = ("neutral_gray", "black", "image_mean")
SELECTOR_LABELS = ("consensus", "native_confidence", "oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        nargs=5,
        metavar=("MODEL", "NEUTRAL", "BLACK", "IMAGE_MEAN", "CONSENSUS"),
        required=True,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"repair report is not an object: {path}")
    return payload


def _metric(summary: dict[str, object], name: str) -> dict[str, object]:
    metric = summary["by_metric"][name]
    if metric["status"] != "available":
        raise ValueError(f"required repair metric is unavailable: {name}")
    aggregate = metric["scene_bootstrap"]["metrics"]
    recovery = metric["gap_recovery"]
    return {
        "identity_error": aggregate["identity_error"]["point_estimate"],
        "corrupt_error": aggregate["corrupt_error"]["point_estimate"],
        "repaired_error": aggregate["repaired_error"]["point_estimate"],
        "clean_relative_degradation": aggregate["clean_relative_degradation"][
            "point_estimate"
        ],
        "gap_recovery": {
            key: recovery[key]
            for key in (
                "point_estimate",
                "lower",
                "upper",
                "valid_replicates",
                "undefined_replicates",
            )
        },
    }


def _record(
    *,
    model: str,
    method: str,
    kind: str,
    summary: dict[str, object],
    compute: dict[str, object],
    peak_vram_bytes: int,
    selection_frequency: dict[str, int] | None,
) -> dict[str, object]:
    return {
        "model": model,
        "method": method,
        "kind": kind,
        "model_runs_per_scene": int(compute["model_runs_per_scene"]),
        "median_model_compute_seconds_per_scene": float(
            compute["median_model_compute_seconds_per_scene"]
        ),
        "maximum_peak_vram_bytes": peak_vram_bytes,
        "selection_frequency": selection_frequency,
        "rotation": _metric(summary, ROTATION),
        "depth": _metric(summary, DEPTH),
    }


def summarize_model(
    model: str,
    fill_paths: dict[str, Path],
    consensus_path: Path,
) -> dict[str, object]:
    fills = {label: _load(path) for label, path in fill_paths.items()}
    consensus = _load(consensus_path)
    if consensus.get("model") != model:
        raise ValueError(f"consensus model mismatch for {model}")
    dataset = consensus["dataset"]
    scenes = consensus["scenes"]
    for label, report in fills.items():
        if (
            report.get("model") != model
            or report.get("dataset") != dataset
            or report.get("scenes") != scenes
        ):
            raise ValueError(f"{label} report design mismatch for {model}")
    if tuple(consensus["candidate_order"]) != FILL_LABELS:
        raise ValueError(f"candidate order drift for {model}")

    candidate_compute = consensus["candidate_compute"]
    method_compute = consensus["method_compute"]
    maximum_peak = max(
        int(candidate_compute[label]["maximum_peak_vram_bytes"])
        for label in FILL_LABELS
    )
    records: list[dict[str, object]] = []
    for label in FILL_LABELS:
        compute = {
            **candidate_compute[label],
            "model_runs_per_scene": 1,
        }
        records.append(
            _record(
                model=model,
                method=label,
                kind="analytic_fill",
                summary=fills[label],
                compute=compute,
                peak_vram_bytes=int(compute["maximum_peak_vram_bytes"]),
                selection_frequency=None,
            )
        )

    analytic = consensus["method_summaries"]["analytic_single_pass"]
    for metric in (ROTATION, DEPTH):
        if _metric(analytic, metric) != _metric(fills["neutral_gray"], metric):
            raise ValueError(f"analytic/neutral summary mismatch for {model}/{metric}")
    for label in SELECTOR_LABELS:
        records.append(
            _record(
                model=model,
                method=label,
                kind="three_fill_selector",
                summary=consensus["method_summaries"][label],
                compute=method_compute[label],
                peak_vram_bytes=maximum_peak,
                selection_frequency={
                    key: int(value)
                    for key, value in consensus["selection_frequencies"][label].items()
                },
            )
        )
    return {
        "model": model,
        "dataset": dataset,
        "scene_count": len(scenes),
        "scenes": scenes,
        "candidate_order": list(FILL_LABELS),
        "protocol_sha256": consensus["protocol_sha256"],
        "source_reports": {
            **{
                label: {
                    "path": str(fill_paths[label]),
                    "sha256": _sha256(fill_paths[label]),
                }
                for label in FILL_LABELS
            },
            "consensus": {
                "path": str(consensus_path),
                "sha256": _sha256(consensus_path),
            },
        },
        "records": records,
    }


def main() -> None:
    args = parse_args()
    models: list[dict[str, object]] = []
    seen: set[str] = set()
    for model, neutral, black, image_mean, consensus in args.model:
        if model in seen:
            raise ValueError(f"duplicate model: {model}")
        seen.add(model)
        models.append(
            summarize_model(
                model,
                {
                    "neutral_gray": Path(neutral),
                    "black": Path(black),
                    "image_mean": Path(image_mean),
                },
                Path(consensus),
            )
        )
    datasets = {record["dataset"] for record in models}
    protocols = {record["protocol_sha256"] for record in models}
    if len(datasets) != 1 or len(protocols) != 1:
        raise ValueError("repair ablation cannot pool dataset or protocol designs")
    payload = {
        "schema_version": "1.0",
        "status": "complete",
        "dataset": next(iter(datasets)),
        "protocol_sha256": next(iter(protocols)),
        "models_are_separate": True,
        "model_count": len(models),
        "models": models,
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
                "model_count": len(models),
                "record_count": sum(len(record["records"]) for record in models),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
