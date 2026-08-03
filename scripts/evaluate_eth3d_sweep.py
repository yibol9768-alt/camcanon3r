#!/usr/bin/env python3
"""Evaluate an ETH3D variant sweep and compute deltas from identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.eth3d import evaluate_eth3d_prediction
from camcanon3r.summary import summarize_eth3d_evaluations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("calibration_dir", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    depth = parser.add_mutually_exclusive_group(required=True)
    depth.add_argument("--depth-dir", type=Path)
    depth.add_argument("--pose-only", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true")
    existing.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executed: list[str] = []
    skipped: list[str] = []
    for variant in args.variants:
        prediction = args.prediction_root / f"{variant}.npz"
        if not prediction.is_file():
            raise FileNotFoundError(f"prediction is missing: {prediction}")
        output = args.result_root / f"{variant}_vs_gt.json"
        if output.exists() and args.resume:
            skipped.append(str(output))
            continue
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"evaluation exists; use --resume or --overwrite: {output}"
            )
        result = evaluate_eth3d_prediction(
            prediction,
            args.calibration_dir,
            depth_dir=None if args.pose_only else args.depth_dir,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        executed.append(str(output))

    evaluation_paths = sorted(args.result_root.glob("*_vs_gt.json"))
    summary = summarize_eth3d_evaluations(
        evaluation_paths,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary_path = args.result_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "executed_count": len(executed),
                "skipped_count": len(skipped),
                "summary": str(summary_path),
            }
        )
    )


if __name__ == "__main__":
    main()
