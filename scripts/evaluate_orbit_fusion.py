#!/usr/bin/env python3
"""Evaluate the full camera-constrained orbit fusion against matched baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from camcanon3r.dtu import evaluate_dtu_prediction
from camcanon3r.eth3d import evaluate_eth3d_prediction
from camcanon3r.orbit_preparation import load_orbit_protocol
from camcanon3r.prediction import write_json_atomic
from camcanon3r.summary import (
    summarize_dtu_evaluations,
    summarize_eth3d_evaluations,
)

_DTU_SCENE = re.compile(r"^scan(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_root", type=Path)
    parser.add_argument("projection_root", type=Path)
    parser.add_argument("identity_prediction_root", type=Path)
    parser.add_argument("analytic_prediction_root", type=Path)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("summary_output", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", choices=("eth3d", "dtu"), required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--model", choices=("vggt", "dust3r"), required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--identity-variant", default="identity")
    parser.add_argument("--analytic-variant", default="canonical_asymmetric_crop_075")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=1701)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fusion_metadata(
    path: Path,
    *,
    protocol_sha256: str,
    scene: str,
) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"full orbit fusion pair is missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != "canonical-orbit-fusion-0.1"
        or metadata.get("method") != "response_fusion"
        or metadata.get("scene") != scene
        or metadata.get("protocol_sha256") != protocol_sha256
        or metadata.get("prediction_sha256") != _sha256(path)
        or metadata.get("camera_only") is not False
        or metadata.get("diagnostics", {}).get("ground_truth_used") is not False
        or metadata.get("diagnostics", {}).get("common_source_support_only") is not True
    ):
        raise ValueError(f"full orbit fusion provenance mismatch: {scene}")
    return metadata


def _prediction_paths(args: argparse.Namespace, scene: str) -> dict[str, Path]:
    return {
        "identity": (
            args.identity_prediction_root / scene / f"{args.identity_variant}.npz"
        ),
        "analytic_repair": (
            args.analytic_prediction_root / scene / f"{args.analytic_variant}.npz"
        ),
        "response_fusion": args.projection_root / scene / "response_fusion.npz",
    }


def _evaluate(
    args: argparse.Namespace,
    *,
    scene: str,
    method: str,
    prediction: Path,
) -> dict[str, Any]:
    if not prediction.is_file() or not prediction.with_suffix(".json").is_file():
        raise FileNotFoundError(
            f"matched fusion evaluation input is missing: {prediction}"
        )
    if args.dataset == "eth3d":
        result = evaluate_eth3d_prediction(
            prediction,
            args.selection_root / scene / "dslr_calibration_jpg",
            depth_dir=(args.selection_root / scene / "ground_truth_depth/dslr_images"),
        )
    else:
        match = _DTU_SCENE.fullmatch(scene)
        if match is None:
            raise ValueError(f"cannot parse DTU scene label: {scene}")
        result = evaluate_dtu_prediction(
            prediction,
            args.selection_root / "calibration/cal18",
            scan=int(match.group(1)),
            gt_root=args.selection_root / "gt",
        )
    result.update(
        {
            "scene": scene,
            "variant": method,
            "dataset": args.dataset_label,
            "model": args.model,
            "prediction_sha256": _sha256(prediction),
        }
    )
    return result


def _validate_existing(
    path: Path,
    *,
    args: argparse.Namespace,
    scene: str,
    method: str,
    prediction: Path,
) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "scene": scene,
        "variant": method,
        "dataset": args.dataset_label,
        "model": args.model,
        "prediction": str(prediction.resolve()),
        "prediction_sha256": _sha256(prediction),
    }
    if {key: record.get(key) for key in expected} != expected:
        raise ValueError(f"existing fusion evaluation changed: {scene}/{method}")


def main() -> None:
    args = parse_args()
    if not args.scenes or len(set(args.scenes)) != len(args.scenes):
        raise ValueError("scenes must be a non-empty unique list")
    load_orbit_protocol(args.protocol)
    protocol_sha256 = _sha256(args.protocol)
    if args.summary_output.exists() and not args.resume:
        raise FileExistsError(
            f"fusion summary exists; use --resume: {args.summary_output}"
        )
    outputs = []
    completed = 0
    skipped = 0
    for scene in args.scenes:
        paths = _prediction_paths(args, scene)
        fusion_metadata = _fusion_metadata(
            paths["response_fusion"],
            protocol_sha256=protocol_sha256,
            scene=scene,
        )
        for method, prediction in paths.items():
            output = args.results_root / scene / f"{method}_vs_gt.json"
            if output.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"fusion evaluation exists; use --resume: {output}"
                    )
                _validate_existing(
                    output,
                    args=args,
                    scene=scene,
                    method=method,
                    prediction=prediction,
                )
                skipped += 1
            else:
                result = _evaluate(
                    args,
                    scene=scene,
                    method=method,
                    prediction=prediction,
                )
                result.update(
                    {
                        "orbit_protocol": str(args.protocol.resolve()),
                        "orbit_protocol_sha256": protocol_sha256,
                        "fusion_prediction_sha256": fusion_metadata[
                            "prediction_sha256"
                        ],
                        "method_ground_truth_used": False,
                    }
                )
                write_json_atomic(output, result)
                completed += 1
                print(
                    json.dumps(
                        {
                            "event": "evaluation_complete",
                            "scene": scene,
                            "method": method,
                        }
                    ),
                    flush=True,
                )
            outputs.append(output)

    summary = (
        summarize_eth3d_evaluations(
            outputs,
            reference_variant="analytic_repair",
            bootstrap_replicates=args.bootstrap_replicates,
            confidence_level=args.confidence_level,
            bootstrap_seed=args.bootstrap_seed,
        )
        if args.dataset == "eth3d"
        else summarize_dtu_evaluations(
            outputs,
            reference_variant="analytic_repair",
            bootstrap_replicates=args.bootstrap_replicates,
            confidence_level=args.confidence_level,
            bootstrap_seed=args.bootstrap_seed,
        )
    )
    report = {
        "schema_version": "canonical-orbit-fusion-evaluation-0.1",
        "status": "complete",
        "dataset": args.dataset_label,
        "dataset_family": args.dataset,
        "model": args.model,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha256,
        "method_ground_truth_used": False,
        "scene_count": len(args.scenes),
        "evaluation_count": len(outputs),
        "completed_count": completed,
        "resumed_count": skipped,
        "summary": summary,
    }
    write_json_atomic(args.summary_output, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "scene_count": len(args.scenes),
                "summary": str(args.summary_output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
