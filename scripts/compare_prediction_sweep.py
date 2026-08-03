#!/usr/bin/env python3
"""Compare a multi-scene prediction sweep with identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.comparison import compare_predictions
from camcanon3r.summary import summarize_comparison_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--rotation-threshold", type=float, default=2.0)
    parser.add_argument("--depth-threshold", type=float, default=0.05)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true")
    existing.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executed: list[str] = []
    skipped: list[str] = []
    for scene in args.scenes:
        reference = args.prediction_root / scene / "identity.npz"
        if not reference.is_file():
            raise FileNotFoundError(f"identity prediction is missing: {reference}")
        for variant in args.variants:
            candidate = args.prediction_root / scene / f"{variant}.npz"
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"candidate prediction is missing: {candidate}"
                )
            output = args.result_root / scene / f"{variant}_vs_identity.json"
            if output.exists() and args.resume:
                skipped.append(str(output))
                continue
            if output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"comparison exists; use --resume or --overwrite: {output}"
                )
            result = compare_predictions(
                reference,
                candidate,
                candidate_label=variant,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            executed.append(str(output))

    comparison_paths = sorted(args.result_root.rglob("*_vs_identity.json"))
    summary = summarize_comparison_files(
        comparison_paths,
        rotation_threshold=args.rotation_threshold,
        depth_threshold=args.depth_threshold,
    )
    summary_path = args.result_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
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
