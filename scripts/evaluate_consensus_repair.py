#!/usr/bin/env python3
"""Evaluate frozen cross-fill consensus repair and matched baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from camcanon3r.repair_consensus import (
    read_evaluation,
    summarize_consensus_repair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("original_results", type=Path)
    parser.add_argument("clean_results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        nargs=4,
        action="append",
        metavar=("LABEL", "VARIANT", "PREDICTIONS", "RESULTS"),
        required=True,
    )
    parser.add_argument("--identity-variant", default="identity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--minimum-gap", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_scenes(args: argparse.Namespace, protocol: dict[str, object]):
    candidate_design = [
        {
            "label": values[0],
            "variant": values[1],
            "predictions": Path(values[2]),
            "results": Path(values[3]),
        }
        for values in args.candidate
    ]
    expected = [
        (str(record["label"]), str(record["repaired_variant"]))
        for record in protocol["candidate_order"]
    ]
    actual = [(record["label"], record["variant"]) for record in candidate_design]
    if actual != expected:
        raise ValueError(
            "repair candidates do not match frozen protocol order: "
            f"expected={expected}, actual={actual}"
        )
    original_scenes = {
        path.name for path in args.original_results.iterdir() if path.is_dir()
    }
    clean_scenes = {path.name for path in args.clean_results.iterdir() if path.is_dir()}
    if not original_scenes or original_scenes != clean_scenes:
        raise ValueError("original and clean-control repair scenes do not match")
    for record in candidate_design:
        prediction_scenes = {
            path.name for path in record["predictions"].iterdir() if path.is_dir()
        }
        result_scenes = {
            path.name for path in record["results"].iterdir() if path.is_dir()
        }
        if prediction_scenes != original_scenes or result_scenes != original_scenes:
            raise ValueError(f"candidate scene design mismatch for {record['label']}")

    corrupt_variant = str(protocol["source_variant"])
    scenes: dict[str, dict[str, object]] = {}
    for scene in sorted(original_scenes):
        candidates = {}
        for record in candidate_design:
            prediction = record["predictions"] / scene / f"{record['variant']}.npz"
            if (
                not prediction.is_file()
                or not prediction.with_suffix(".json").is_file()
            ):
                raise FileNotFoundError(
                    f"complete repair prediction is missing: {prediction}"
                )
            candidates[record["label"]] = (
                prediction,
                read_evaluation(
                    record["results"] / scene / f"{record['variant']}_vs_gt.json",
                    scene=scene,
                    variant=record["variant"],
                ),
            )
        scenes[scene] = {
            "identity": read_evaluation(
                args.original_results / scene / f"{args.identity_variant}_vs_gt.json",
                scene=scene,
                variant=args.identity_variant,
            ),
            "corrupt": read_evaluation(
                args.original_results / scene / f"{corrupt_variant}_vs_gt.json",
                scene=scene,
                variant=corrupt_variant,
            ),
            "clean_control": read_evaluation(
                args.clean_results / scene / f"{args.identity_variant}_vs_gt.json",
                scene=scene,
                variant=args.identity_variant,
            ),
            "candidates": candidates,
        }
    return scenes, candidate_design


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not protocol.get("frozen_before_repair_ground_truth_results"):
        raise ValueError("repair consensus protocol was not frozen before results")
    scenes, candidate_design = _load_scenes(args, protocol)
    promotion = protocol["promotion"]
    summary = summarize_consensus_repair(
        scenes,
        candidate_order=[str(record["label"]) for record in candidate_design],
        minimum_gap=args.minimum_gap,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
        recovery_threshold=float(promotion["minimum_gap_recovery"]),
        clean_relative_threshold=float(promotion["maximum_clean_relative_degradation"]),
    )
    summary.update(
        {
            "model": args.model,
            "dataset": args.dataset,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": _sha256(args.protocol),
            "original_results": str(args.original_results.resolve()),
            "clean_results": str(args.clean_results.resolve()),
            "candidate_roots": [
                {
                    "label": record["label"],
                    "variant": record["variant"],
                    "predictions": str(record["predictions"].resolve()),
                    "results": str(record["results"].resolve()),
                }
                for record in candidate_design
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "scene_count": summary["scene_count"],
                "promotion": summary["promotion"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
