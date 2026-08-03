#!/usr/bin/env python3
"""Compare two VGGT camera predictions using gauge-invariant pairwise poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from camcanon3r.metrics import aligned_depth_consistency, pairwise_relative_pose_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-label", default="identity")
    parser.add_argument("--candidate-label")
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0, "median": None, "mean": None, "p90": None}
    return {
        "count": len(finite),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def finite_or_none(value: np.floating) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    args = parse_args()
    with np.load(args.reference) as reference_data:
        reference = reference_data["extrinsic"]
        reference_depth = reference_data["depth"]
        reference_affine = reference_data["source_to_model_affine"]
    with np.load(args.candidate) as candidate_data:
        candidate = candidate_data["extrinsic"]
        candidate_depth = candidate_data["depth"]
        candidate_affine = candidate_data["source_to_model_affine"]

    errors = pairwise_relative_pose_errors(reference, candidate)
    depth = aligned_depth_consistency(
        reference_depth,
        candidate_depth,
        reference_affine,
        candidate_affine,
    )
    pairs = errors["pairs"]
    result = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label or args.candidate.stem,
        "view_count": int(max(pairs.reshape(-1)) + 1),
        "pair_count": len(pairs),
        "rotation_degrees": summarize(errors["rotation_degrees"]),
        "translation_direction_degrees": summarize(
            errors["translation_direction_degrees"]
        ),
        "aligned_depth_consistency": depth,
        "per_pair": [
            {
                "views": [int(first), int(second)],
                "rotation_degrees": finite_or_none(rotation),
                "translation_direction_degrees": finite_or_none(translation),
            }
            for (first, second), rotation, translation in zip(
                pairs,
                errors["rotation_degrees"],
                errors["translation_direction_degrees"],
                strict=True,
            )
        ],
    }
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
