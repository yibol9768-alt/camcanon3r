#!/usr/bin/env python3
"""Aggregate a paired multi-scene analytic-repair experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.repair_evaluation import summarize_repair_evaluations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("original_results", type=Path)
    parser.add_argument("repaired_results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--identity-variant", default="identity")
    parser.add_argument("--corrupt-variant", required=True)
    parser.add_argument("--clean-control-variant", default="identity")
    parser.add_argument("--repaired-variant", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--minimum-gap", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--recovery-threshold", type=float, default=0.30)
    parser.add_argument("--clean-relative-threshold", type=float, default=0.02)
    return parser.parse_args()


def _read_evaluation(path: Path, *, scene: str, variant: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"repair evaluation is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("scene") != scene or record.get("variant") != variant:
        raise ValueError(
            f"repair evaluation identity mismatch for {path}: "
            f"scene={record.get('scene')!r}, variant={record.get('variant')!r}"
        )
    return record


def _load_scene_records(
    original_results: Path,
    repaired_results: Path,
    *,
    identity_variant: str,
    corrupt_variant: str,
    clean_control_variant: str,
    repaired_variant: str,
) -> dict[
    str,
    tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ],
]:
    original_scenes = {
        path.name for path in original_results.iterdir() if path.is_dir()
    }
    repaired_scenes = {
        path.name for path in repaired_results.iterdir() if path.is_dir()
    }
    if not original_scenes or original_scenes != repaired_scenes:
        raise ValueError(
            "repair scene design mismatch: "
            f"original={sorted(original_scenes)}, repaired={sorted(repaired_scenes)}"
        )
    scene_records = {}
    for scene in sorted(original_scenes):
        original_scene = original_results / scene
        repaired_scene = repaired_results / scene
        scene_records[scene] = (
            _read_evaluation(
                original_scene / f"{identity_variant}_vs_gt.json",
                scene=scene,
                variant=identity_variant,
            ),
            _read_evaluation(
                original_scene / f"{corrupt_variant}_vs_gt.json",
                scene=scene,
                variant=corrupt_variant,
            ),
            _read_evaluation(
                repaired_scene / f"{repaired_variant}_vs_gt.json",
                scene=scene,
                variant=repaired_variant,
            ),
            _read_evaluation(
                repaired_scene / f"{clean_control_variant}_vs_gt.json",
                scene=scene,
                variant=clean_control_variant,
            ),
        )
    return scene_records


def main() -> None:
    args = parse_args()
    scene_records = _load_scene_records(
        args.original_results,
        args.repaired_results,
        identity_variant=args.identity_variant,
        corrupt_variant=args.corrupt_variant,
        clean_control_variant=args.clean_control_variant,
        repaired_variant=args.repaired_variant,
    )
    summary = summarize_repair_evaluations(
        scene_records,
        minimum_gap=args.minimum_gap,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
        recovery_threshold=args.recovery_threshold,
        clean_relative_threshold=args.clean_relative_threshold,
    )
    summary.update(
        {
            "model": args.model,
            "dataset": args.dataset,
            "original_results": str(args.original_results.resolve()),
            "repaired_results": str(args.repaired_results.resolve()),
            "identity_variant": args.identity_variant,
            "corrupt_variant": args.corrupt_variant,
            "clean_control_variant": args.clean_control_variant,
            "repaired_variant": args.repaired_variant,
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
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
