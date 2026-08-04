#!/usr/bin/env python3
"""Prepare canonical-camera repair inputs for a multi-scene sweep."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median

from camcanon3r.protocol import list_images
from camcanon3r.repair import FILL_POLICIES, canonicalize_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--prefix", default="canonical_")
    parser.add_argument(
        "--fill-policy", choices=sorted(FILL_POLICIES), default="neutral_gray"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="atomically checkpoint per-scene canonicalization wall time",
    )
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true")
    existing.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _output_name(variant: str, prefix: str) -> str:
    return "identity" if variant == "identity" else f"{prefix}{variant}"


def _is_complete(source: Path, output: Path) -> bool:
    if not output.is_dir() or not (output.parent / "manifest.json").is_file():
        return False
    source_names = {path.name for path in list_images(source)}
    output_names = {path.name for path in list_images(output)}
    mask_dir = output.parent / "_masks" / output.name
    mask_names = {path.name for path in mask_dir.glob("*.png")}
    return source_names == output_names == mask_names


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _report_design(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "prepared_root": str(args.prepared_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "scenes": args.scenes,
        "source_variants": args.variants,
        "prefix": args.prefix,
        "fill_policy": args.fill_policy,
        "timing_boundary": (
            "source image decode, inverse affine image and mask warps, PNG writes, "
            "and repair manifest update; excludes later model inference"
        ),
    }


def _load_report(args: argparse.Namespace) -> dict[str, object] | None:
    if args.report is None or not args.report.exists() or args.overwrite:
        return None
    if not args.resume:
        raise FileExistsError(
            f"canonicalization report exists; use --resume or --overwrite: {args.report}"
        )
    report = json.loads(args.report.read_text(encoding="utf-8"))
    design = _report_design(args)
    actual = {key: report.get(key) for key in design}
    if actual != design:
        raise ValueError(
            "canonicalization timing report design mismatch: "
            f"expected={design}, actual={actual}"
        )
    records = report.get("records")
    if not isinstance(records, list):
        raise TypeError("canonicalization timing report records are invalid")
    return report


def _checkpoint_report(
    args: argparse.Namespace,
    records: list[dict[str, object]],
    *,
    complete: bool,
) -> None:
    if args.report is None:
        return
    payload = {
        **_report_design(args),
        "status": "complete" if complete else "in_progress",
        "record_count": len(records),
        "canonicalization_seconds": {
            "count": len(records),
            "total": sum(
                float(record["canonicalization_seconds"]) for record in records
            ),
            "median": (
                float(
                    median(
                        float(record["canonicalization_seconds"]) for record in records
                    )
                )
                if records
                else None
            ),
            "maximum": (
                max(float(record["canonicalization_seconds"]) for record in records)
                if records
                else None
            ),
        },
        "records": records,
    }
    _write_json_atomic(args.report, payload)


def main() -> None:
    args = parse_args()
    existing_report = _load_report(args)
    timing_records = list(existing_report["records"]) if existing_report else []
    timing_by_job = {
        (str(record["scene"]), str(record["source_variant"])): record
        for record in timing_records
    }
    if len(timing_by_job) != len(timing_records):
        raise ValueError("canonicalization report contains duplicate jobs")
    for (scene, variant), record in timing_by_job.items():
        elapsed = float(record["canonicalization_seconds"])
        if (
            not math.isfinite(elapsed)
            or elapsed < 0.0
            or record.get("output_variant") != _output_name(variant, args.prefix)
            or int(record.get("image_count", 0)) <= 0
        ):
            raise ValueError(
                f"canonicalization timing record is invalid: {scene}/{variant}"
            )
    expected_jobs = {
        (str(scene), str(variant)) for scene in args.scenes for variant in args.variants
    }
    if set(timing_by_job) - expected_jobs:
        raise ValueError("canonicalization report contains jobs outside the design")
    executed: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for scene in args.scenes:
        for variant in args.variants:
            source = args.prepared_root / scene / variant
            output = args.output_root / scene / _output_name(variant, args.prefix)
            complete = _is_complete(source, output) if output.exists() else False
            if complete and args.resume:
                if args.report is not None and (scene, variant) not in timing_by_job:
                    raise RuntimeError(
                        "complete repair output has no resumable timing record: "
                        f"{scene}/{variant}"
                    )
                skipped.append({"scene": scene, "variant": variant})
                continue
            if output.exists() and not args.overwrite:
                state = "complete" if complete else "partial"
                raise FileExistsError(
                    f"{state} repair output requires --resume or --overwrite: {output}"
                )
            start = time.perf_counter()
            result = canonicalize_variant(source, output, fill_policy=args.fill_policy)
            elapsed = time.perf_counter() - start
            valid_fractions = [
                item["repair"]["valid_fraction"] for item in result["images"]
            ]
            executed.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "output_variant": output.name,
                    "canonicalization_seconds": elapsed,
                    "mean_valid_fraction": sum(valid_fractions) / len(valid_fractions),
                }
            )
            timing_record = {
                "scene": scene,
                "source_variant": variant,
                "output_variant": output.name,
                "image_count": len(result["images"]),
                "canonicalization_seconds": elapsed,
            }
            timing_by_job[(scene, variant)] = timing_record
            timing_records = [
                timing_by_job[key]
                for key in (
                    (design_scene, design_variant)
                    for design_scene in args.scenes
                    for design_variant in args.variants
                )
                if key in timing_by_job
            ]
            _checkpoint_report(args, timing_records, complete=False)
    if set(timing_by_job) != expected_jobs and args.report is not None:
        raise RuntimeError("canonicalization timing report is incomplete")
    _checkpoint_report(args, timing_records, complete=True)
    print(
        json.dumps(
            {
                "executed_count": len(executed),
                "skipped_count": len(skipped),
                "executed": executed,
                "skipped": skipped,
            }
        )
    )


if __name__ == "__main__":
    main()
