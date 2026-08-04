#!/usr/bin/env python3
"""Evaluate a disagreement or native-uncertainty score from case records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from camcanon3r.reliability import reliability_summary, resolve_case_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--failure-threshold", type=float, required=True)
    parser.add_argument("--scene-field", default="scene")
    parser.add_argument("--error-field", default="error")
    parser.add_argument("--uncertainty-field", default="uncertainty")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_cases(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cases")
    if not isinstance(payload, list):
        raise TypeError("JSON input must be a list or an object with a cases list")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    records = _read_cases(args.cases)
    if not records:
        raise ValueError("at least one reliability case is required")
    result = reliability_summary(
        [float(resolve_case_field(row, args.error_field)) for row in records],
        [
            float(resolve_case_field(row, args.uncertainty_field))
            for row in records
        ],
        scenes=[
            str(resolve_case_field(row, args.scene_field)) for row in records
        ],
        failure_threshold=args.failure_threshold,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["input"] = {
        "cases": str(args.cases),
        "cases_sha256": _sha256_file(args.cases),
        "scene_field": args.scene_field,
        "error_field": args.error_field,
        "uncertainty_field": args.uncertainty_field,
    }
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
