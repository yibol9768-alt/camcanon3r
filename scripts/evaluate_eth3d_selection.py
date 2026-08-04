#!/usr/bin/env python3
"""Evaluate a frozen multi-scene ETH3D selection and summarize it atomically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.eth3d import evaluate_eth3d_prediction
from camcanon3r.summary import summarize_eth3d_evaluations

EVALUATION_PROTOCOL_VERSION = "2.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_root", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--domain", choices=("raw", "undistorted"), required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def _load_selection(selection_root: Path) -> dict[str, object]:
    report_path = selection_root / "selection_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("completed_at") is None:
        raise RuntimeError("ETH3D selection report is not complete")
    selection = report.get("selection")
    if not isinstance(selection, dict) or not isinstance(
        selection.get("scenes"), list
    ):
        raise TypeError("ETH3D selection report has no scene selection")
    return selection


def _expected_jobs(
    selection_root: Path,
    prediction_root: Path,
    results_root: Path,
    *,
    domain: str,
    variants: list[str],
) -> list[dict[str, object]]:
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be a non-empty list without duplicates")
    selection = _load_selection(selection_root)
    jobs: list[dict[str, object]] = []
    seen_scenes: set[str] = set()
    for scene_record in selection["scenes"]:
        scene = str(scene_record["scene"])
        if scene in seen_scenes:
            raise ValueError(f"duplicate scene in ETH3D selection: {scene}")
        seen_scenes.add(scene)
        expected_inputs = [str(name) for name in scene_record["image_names"]]
        calibration_name = (
            "dslr_calibration_jpg"
            if domain == "raw"
            else "dslr_calibration_undistorted"
        )
        calibration_dir = selection_root / scene / calibration_name
        depth_dir = (
            selection_root / scene / "ground_truth_depth/dslr_images"
            if domain == "raw"
            else None
        )
        if not calibration_dir.is_dir():
            raise FileNotFoundError(f"calibration directory is missing: {calibration_dir}")
        if depth_dir is not None and not depth_dir.is_dir():
            raise FileNotFoundError(f"depth directory is missing: {depth_dir}")
        for variant in variants:
            prediction = prediction_root / scene / f"{variant}.npz"
            metadata = prediction.with_suffix(".json")
            if not prediction.is_file() or not metadata.is_file():
                raise FileNotFoundError(
                    f"complete prediction is missing: {prediction} and {metadata}"
                )
            prediction_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            actual_stems = [
                Path(str(name)).stem for name in prediction_metadata.get("inputs", [])
            ]
            expected_stems = [Path(name).stem for name in expected_inputs]
            if actual_stems != expected_stems:
                raise RuntimeError(
                    f"prediction inputs do not match frozen selection for {scene}/"
                    f"{variant}: expected={expected_stems}, actual={actual_stems}"
                )
            jobs.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "domain": domain,
                    "prediction": prediction,
                    "calibration_dir": calibration_dir,
                    "depth_dir": depth_dir,
                    "output": results_root / scene / f"{variant}_vs_gt.json",
                }
            )
    return jobs


def _validate_existing(record: dict[str, object], job: dict[str, object]) -> None:
    expected = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "scene": job["scene"],
        "variant": job["variant"],
        "domain": job["domain"],
        "prediction": str(Path(job["prediction"]).resolve()),
        "calibration_dir": str(Path(job["calibration_dir"]).resolve()),
        "depth_dir": (
            str(Path(job["depth_dir"]).resolve()) if job["depth_dir"] else None
        ),
    }
    actual = {key: record.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"existing ETH3D evaluation does not match frozen job: "
            f"expected={expected}, actual={actual}"
        )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    jobs = _expected_jobs(
        args.selection_root,
        args.prediction_root,
        args.results_root,
        domain=args.domain,
        variants=args.variants,
    )
    expected_outputs = {Path(job["output"]).resolve() for job in jobs}
    unexpected_outputs = {
        path.resolve() for path in args.results_root.rglob("*_vs_gt.json")
    } - expected_outputs
    if unexpected_outputs:
        raise RuntimeError(
            "results root contains evaluations outside the frozen design: "
            f"{sorted(str(path) for path in unexpected_outputs)}"
        )

    completed = 0
    skipped = 0
    for job in jobs:
        output = Path(job["output"])
        if output.exists():
            if not args.resume:
                raise FileExistsError(f"evaluation exists; use --resume: {output}")
            existing = json.loads(output.read_text(encoding="utf-8"))
            _validate_existing(existing, job)
            skipped += 1
            continue
        result = evaluate_eth3d_prediction(
            Path(job["prediction"]),
            Path(job["calibration_dir"]),
            depth_dir=(Path(job["depth_dir"]) if job["depth_dir"] else None),
        )
        result.update(
            {
                "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
                "scene": job["scene"],
                "variant": job["variant"],
                "domain": job["domain"],
                "calibration_dir": str(Path(job["calibration_dir"]).resolve()),
                "depth_dir": (
                    str(Path(job["depth_dir"]).resolve())
                    if job["depth_dir"]
                    else None
                ),
            }
        )
        _write_json_atomic(output, result)
        completed += 1
        print(
            json.dumps(
                {
                    "scene": job["scene"],
                    "variant": job["variant"],
                    "domain": job["domain"],
                    "status": "complete",
                }
            ),
            flush=True,
        )

    outputs = sorted(Path(job["output"]) for job in jobs)
    summary = summarize_eth3d_evaluations(
        outputs,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "domain": args.domain,
            "selection_report": str(
                (args.selection_root / "selection_report.json").resolve()
            ),
            "prediction_root": str(args.prediction_root.resolve()),
        }
    )
    summary_output = args.summary_output or args.results_root / "summary.json"
    _write_json_atomic(summary_output, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "domain": args.domain,
                "evaluation_count": len(jobs),
                "completed_count": completed,
                "skipped_count": skipped,
                "summary": str(summary_output),
            }
        )
    )


if __name__ == "__main__":
    main()
