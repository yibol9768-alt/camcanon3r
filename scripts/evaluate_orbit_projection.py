#!/usr/bin/env python3
"""Evaluate camera-only orbit projections and matched selection baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from camcanon3r.dtu import read_dtu_projection
from camcanon3r.eth3d import read_colmap_images
from camcanon3r.orbit_evaluation import (
    evaluate_camera_extrinsics,
    select_ground_truth_oracle,
    summarize_orbit_camera_evaluations,
)
from camcanon3r.orbit_preparation import load_orbit_protocol
from camcanon3r.prediction import write_json_atomic

_DTU_CAMERA = re.compile(r"^rect_(\d{3})_\d+_r5000$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_root", type=Path)
    parser.add_argument("orbit_prediction_root", type=Path)
    parser.add_argument("projection_root", type=Path)
    parser.add_argument("identity_prediction_root", type=Path)
    parser.add_argument("analytic_prediction_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", choices=("eth3d", "dtu"), required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--model", choices=("vggt", "dust3r"), required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--identity-variant", default="identity")
    parser.add_argument("--analytic-variant", default="canonical_asymmetric_crop_075")
    parser.add_argument("--eth3d-domain", choices=("raw", "undistorted"), default="raw")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_extrinsic(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"camera prediction is missing: {path}")
    with np.load(path, allow_pickle=False) as prediction:
        if "extrinsic" not in prediction:
            raise ValueError(f"camera prediction has no extrinsic array: {path}")
        return np.asarray(prediction["extrinsic"], dtype=np.float64)


def _prediction_inputs(path: Path) -> list[str]:
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    inputs = [str(value) for value in metadata.get("inputs", [])]
    if len(inputs) < 2:
        raise ValueError(f"prediction input metadata is incomplete: {metadata_path}")
    return inputs


def _target_extrinsics(
    args: argparse.Namespace, scene: str, inputs: list[str]
) -> tuple[np.ndarray, str]:
    if args.dataset == "eth3d":
        calibration_name = (
            "dslr_calibration_jpg"
            if args.eth3d_domain == "raw"
            else "dslr_calibration_undistorted"
        )
        calibration_dir = args.selection_root / scene / calibration_name
        images = read_colmap_images(calibration_dir / "images.txt")
        try:
            target = np.stack([images[Path(name).stem].extrinsic for name in inputs])
        except KeyError as error:
            raise ValueError(
                f"ETH3D input has no matching calibration image: {scene}/{error}"
            ) from error
        return target, str(calibration_dir.resolve())
    calibration_dir = args.selection_root / "calibration/cal18"
    camera_ids = []
    for name in inputs:
        match = _DTU_CAMERA.fullmatch(Path(name).stem)
        if match is None:
            raise ValueError(f"cannot parse DTU camera ID: {scene}/{name}")
        camera_ids.append(int(match.group(1)))
    target = np.stack(
        [
            read_dtu_projection(calibration_dir / f"pos_{camera_id:03d}.txt")[1]
            for camera_id in camera_ids
        ]
    )
    return target, str(calibration_dir.resolve())


def _check_input_stems(path: Path, expected: list[str]) -> None:
    actual = _prediction_inputs(path)
    if [Path(value).stem for value in actual] != [
        Path(value).stem for value in expected
    ]:
        raise ValueError(f"prediction input order does not match orbit: {path}")


def _projection_metadata(
    path: Path, *, protocol_sha256: str, scene: str, method: str
) -> dict[str, Any]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != "canonical-orbit-projection-0.1"
        or metadata.get("protocol_sha256") != protocol_sha256
        or metadata.get("scene") != scene
        or metadata.get("method") != method
        or metadata.get("prediction_sha256") != _sha256(path)
        or metadata.get("camera_only") is not True
    ):
        raise ValueError(f"projected camera provenance mismatch: {scene}/{method}")
    return metadata


def main() -> None:
    args = parse_args()
    if not args.scenes or len(set(args.scenes)) != len(args.scenes):
        raise ValueError("scenes must be a non-empty unique list")
    if args.output.exists():
        raise FileExistsError(f"orbit evaluation already exists: {args.output}")
    protocol = load_orbit_protocol(args.protocol)
    protocol_sha256 = _sha256(args.protocol)
    labels = [str(record["label"]) for record in protocol["orbit"]["ordered_members"]]
    per_scene: dict[str, dict[str, Any]] = {}
    for scene in args.scenes:
        robust_path = args.projection_root / scene / "robust_projection.npz"
        uniform_path = args.projection_root / scene / "uniform_projection.npz"
        robust_metadata = _projection_metadata(
            robust_path,
            protocol_sha256=protocol_sha256,
            scene=scene,
            method="robust_projection",
        )
        uniform_metadata = _projection_metadata(
            uniform_path,
            protocol_sha256=protocol_sha256,
            scene=scene,
            method="uniform_projection",
        )
        inputs = [str(value) for value in robust_metadata["inputs"]]
        if uniform_metadata["inputs"] != inputs:
            raise ValueError(f"projection input order mismatch: {scene}")
        target, calibration = _target_extrinsics(args, scene, inputs)

        identity_path = (
            args.identity_prediction_root / scene / f"{args.identity_variant}.npz"
        )
        analytic_path = (
            args.analytic_prediction_root / scene / f"{args.analytic_variant}.npz"
        )
        _check_input_stems(identity_path, inputs)
        _check_input_stems(analytic_path, inputs)
        members = {
            label: _load_extrinsic(
                args.orbit_prediction_root / scene / f"orbit_{label}.npz"
            )
            for label in labels
        }
        for label in labels:
            _check_input_stems(
                args.orbit_prediction_root / scene / f"orbit_{label}.npz", inputs
            )
        medoid_label = str(robust_metadata["diagnostics"]["orbit_medoid"])
        native_record = max(
            robust_metadata["sources"],
            key=lambda record: (
                float(record["native_confidence_median"]),
                -labels.index(str(record["label"])),
            ),
        )
        native_label = str(native_record["label"])
        oracle_label, oracle_errors = select_ground_truth_oracle(
            target, members, member_order=labels
        )
        method_predictions = {
            "identity": (_load_extrinsic(identity_path), True),
            "analytic_repair": (_load_extrinsic(analytic_path), True),
            "robust_projection": (
                _load_extrinsic(robust_path),
                robust_metadata["diagnostics"]["translation_status"] == "available",
            ),
            "uniform_projection": (
                _load_extrinsic(uniform_path),
                uniform_metadata["diagnostics"]["translation_status"] == "available",
            ),
            "orbit_medoid": (members[medoid_label], True),
            "native_confidence": (members[native_label], True),
            "oracle": (members[oracle_label], True),
        }
        scene_record = {
            method: evaluate_camera_extrinsics(
                target, predicted, translation_available=translation_available
            )
            for method, (predicted, translation_available) in method_predictions.items()
        }
        scene_record["provenance"] = {
            "calibration": calibration,
            "inputs": inputs,
            "identity_prediction": str(identity_path.resolve()),
            "analytic_prediction": str(analytic_path.resolve()),
            "robust_prediction": str(robust_path.resolve()),
            "uniform_prediction": str(uniform_path.resolve()),
            "orbit_medoid_label": medoid_label,
            "native_confidence_label": native_label,
            "oracle_label": oracle_label,
            "oracle_member_rotation_degrees": oracle_errors,
            "ground_truth_used_by_oracle_only": True,
        }
        per_scene[scene] = scene_record
        print(
            json.dumps(
                {
                    "event": "scene_complete",
                    "scene": scene,
                    "robust_rotation_degrees": scene_record["robust_projection"][
                        "relative_rotation_degrees"
                    ]["median"],
                }
            ),
            flush=True,
        )

    promotion = protocol["multi_run_promotion"]
    summary = summarize_orbit_camera_evaluations(
        per_scene,
        minimum_residual_gap_reduction=float(
            promotion["minimum_residual_gap_reduction_vs_one_pass"]
        ),
        maximum_median_error_increase_degrees=float(
            promotion["maximum_median_error_increase_degrees"]
        ),
        bootstrap_replicates=int(promotion["scene_cluster_bootstrap_replicates"]),
        confidence_level=float(promotion["confidence_level"]),
        bootstrap_seed=int(promotion["bootstrap_seed"]),
    )
    report = {
        "schema_version": "canonical-orbit-ground-truth-evaluation-0.1",
        "status": "complete",
        "model": args.model,
        "dataset": args.dataset_label,
        "dataset_family": args.dataset,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha256,
        "historical_evidence_was_known_when_designed": True,
        "projected_ground_truth_result_was_unknown_when_protocol_frozen": True,
        "method_ground_truth_used": False,
        "oracle_ground_truth_used": True,
        "per_scene": per_scene,
        "summary": summary,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "model": args.model,
                "dataset": args.dataset_label,
                "promotion_pass": summary["promotion"]["promotion_pass"],
                "output": str(args.output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
