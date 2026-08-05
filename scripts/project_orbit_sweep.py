#!/usr/bin/env python3
"""Project completed orbit predictions into robust and uniform camera graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from camcanon3r.orbit_preparation import load_orbit_protocol
from camcanon3r.orbit_projection import (
    project_camera_orbit,
    project_camera_response_field,
)
from camcanon3r.prediction import save_npz_compressed_atomic, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _confidence(path: Path) -> tuple[str, float]:
    with np.load(path, allow_pickle=False) as prediction:
        for field in ("world_points_conf", "depth_conf"):
            if field not in prediction:
                continue
            values = np.asarray(prediction[field], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if len(finite):
                return field, float(np.median(finite))
    raise ValueError(f"orbit prediction has no finite native confidence: {path}")


def _source_records(
    prediction_root: Path,
    scene: str,
    labels: list[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    members: dict[str, np.ndarray] = {}
    records = []
    common_inputs: list[str] | None = None
    for label in labels:
        path = prediction_root / scene / f"orbit_{label}.npz"
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"complete orbit prediction is missing: {path} and {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        inputs = [str(value) for value in metadata.get("inputs", [])]
        if len(inputs) < 2:
            raise ValueError(f"orbit prediction has fewer than two inputs: {path}")
        if common_inputs is None:
            common_inputs = inputs
        elif inputs != common_inputs:
            raise ValueError(f"orbit prediction input order differs: {scene}/{label}")
        with np.load(path, allow_pickle=False) as prediction:
            if "extrinsic" not in prediction:
                raise ValueError(f"orbit prediction has no extrinsic array: {path}")
            members[label] = np.asarray(prediction["extrinsic"], dtype=np.float64)
        confidence_field, confidence = _confidence(path)
        records.append(
            {
                "label": label,
                "prediction": str(path.resolve()),
                "prediction_sha256": _sha256(path),
                "metadata": str(metadata_path.resolve()),
                "metadata_sha256": _sha256(metadata_path),
                "native_confidence_field": confidence_field,
                "native_confidence_median": confidence,
            }
        )
    assert common_inputs is not None
    return members, records, common_inputs


def _diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"extrinsic", "rotation", "camera_center"}
    }


def _write_projection(
    path: Path,
    *,
    method: str,
    scene: str,
    result: dict[str, Any],
    inputs: list[str],
    sources: list[dict[str, Any]],
    protocol_path: Path,
    protocol_sha256: str,
    projection_seconds: float,
) -> dict[str, Any]:
    save_npz_compressed_atomic(
        path,
        extrinsic=result["extrinsic"],
        rotation=result["rotation"],
        camera_center=result["camera_center"],
    )
    record = {
        "schema_version": "canonical-orbit-projection-0.1",
        "method": method,
        "scene": scene,
        "camera_only": True,
        "inputs": inputs,
        "prediction": str(path.resolve()),
        "prediction_sha256": _sha256(path),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha256,
        "projection_seconds": projection_seconds,
        "sources": sources,
        "diagnostics": _diagnostics(result),
    }
    write_json_atomic(path.with_suffix(".json"), record)
    return record


def _validate_resumed(
    output_root: Path,
    scene: str,
    sources: list[dict[str, Any]],
    protocol_sha256: str,
) -> dict[str, Any]:
    records = {}
    for method in (
        "response_projection",
        "robust_projection",
        "uniform_projection",
    ):
        path = output_root / scene / f"{method}.npz"
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"resumed projection pair is incomplete: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("method") != method
            or metadata.get("scene") != scene
            or metadata.get("protocol_sha256") != protocol_sha256
            or metadata.get("sources") != sources
            or metadata.get("prediction_sha256") != _sha256(path)
        ):
            raise ValueError(f"resumed projection provenance changed: {scene}/{method}")
        records[method] = metadata
    response = records["response_projection"]
    robust = records["robust_projection"]
    diagnostics = robust["diagnostics"]
    return {
        "scene": scene,
        "response_projection": response["prediction"],
        "response_projection_sha256": response["prediction_sha256"],
        "robust_projection": robust["prediction"],
        "robust_projection_sha256": robust["prediction_sha256"],
        "uniform_projection": records["uniform_projection"]["prediction"],
        "uniform_projection_sha256": records["uniform_projection"]["prediction_sha256"],
        "orbit_medoid": diagnostics["orbit_medoid"],
        "native_confidence": max(
            sources,
            key=lambda record: (
                float(record["native_confidence_median"]),
                -next(
                    index
                    for index, source in enumerate(sources)
                    if source["label"] == record["label"]
                ),
            ),
        )["label"],
        "selected_response_basis": response["diagnostics"]["selected_basis"],
        "translation_status": response["diagnostics"]["translation_status"],
        "projection_seconds": float(response["projection_seconds"])
        + float(robust["projection_seconds"])
        + float(records["uniform_projection"]["projection_seconds"]),
    }


def _design(args: argparse.Namespace, protocol_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "canonical-orbit-projection-sweep-0.1",
        "prediction_root": str(args.prediction_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha256,
        "scenes": args.scenes,
    }


def _checkpoint(
    args: argparse.Namespace,
    protocol_sha256: str,
    records: list[dict[str, Any]],
    *,
    complete: bool,
) -> None:
    durations = [float(record["projection_seconds"]) for record in records]
    write_json_atomic(
        args.report,
        {
            **_design(args, protocol_sha256),
            "status": "complete" if complete else "in_progress",
            "record_count": len(records),
            "projection_seconds": {
                "count": len(durations),
                "total": float(sum(durations)),
                "median": float(median(durations)) if durations else None,
            },
            "records": records,
        },
    )


def main() -> None:
    args = parse_args()
    if not args.scenes or len(set(args.scenes)) != len(args.scenes):
        raise ValueError("scenes must be a non-empty unique list")
    protocol = load_orbit_protocol(args.protocol)
    protocol_sha256 = _sha256(args.protocol)
    labels = [str(record["label"]) for record in protocol["orbit"]["ordered_members"]]
    inverse_pairs = {
        str(record["label"]): str(record["inverse_pair"])
        for record in protocol["orbit"]["ordered_members"]
    }
    placements = {
        str(record["label"]): [float(value) for value in record["placement"]]
        for record in protocol["orbit"]["ordered_members"]
    }
    records: list[dict[str, Any]] = []
    existing: dict[str, dict[str, Any]] = {}
    if args.report.exists():
        if not args.resume:
            raise FileExistsError(
                f"projection report exists; use --resume: {args.report}"
            )
        report = json.loads(args.report.read_text(encoding="utf-8"))
        design = _design(args, protocol_sha256)
        if {key: report.get(key) for key in design} != design:
            raise ValueError("existing projection report design does not match")
        existing = {str(record["scene"]): record for record in report["records"]}

    for scene in args.scenes:
        members, sources, inputs = _source_records(args.prediction_root, scene, labels)
        if scene in existing:
            record = _validate_resumed(
                args.output_root, scene, sources, protocol_sha256
            )
            if record != existing[scene]:
                raise ValueError(f"resumed projection record changed: {scene}")
            records.append(record)
            continue
        start = time.perf_counter()
        response = project_camera_response_field(
            members,
            placements=placements,
            member_order=labels,
            inverse_pairs=inverse_pairs,
            candidate_bases=protocol["response_field"]["candidate_bases"],
            minimum_cv_improvement=float(
                protocol["response_field"][
                    "minimum_relative_cv_improvement_for_more_complex_basis"
                ]
            ),
            ridge=float(protocol["response_field"]["ridge"]),
            tuning_constant=float(protocol["response_field"]["tuning_constant"]),
            scale_floor_degrees=float(
                protocol["response_field"]["robust_scale_floor_degrees"]
            ),
            minimum_effective_members=int(
                protocol["response_field"]["minimum_effective_members"]
            ),
            center_anchor_minimum_weight=float(
                protocol["response_field"]["center_anchor_minimum_weight"]
            ),
            maximum_anchor_deviation_degrees=float(
                protocol["response_field"]["maximum_response_anchor_deviation_degrees"]
            ),
        )
        response_seconds = time.perf_counter() - start
        start = time.perf_counter()
        robust = project_camera_orbit(
            members,
            member_order=labels,
            inverse_pairs=inverse_pairs,
            robust=True,
            tuning_constant=float(protocol["projection"]["tuning_constant"]),
            scale_floor_degrees=float(
                protocol["projection"]["robust_scale_floor_degrees"]
            ),
            minimum_effective_groups=int(
                protocol["projection"]["minimum_effective_groups"]
            ),
        )
        robust_seconds = time.perf_counter() - start
        start = time.perf_counter()
        uniform = project_camera_orbit(
            members,
            member_order=labels,
            inverse_pairs=inverse_pairs,
            robust=False,
        )
        uniform_seconds = time.perf_counter() - start
        scene_root = args.output_root / scene
        response_metadata = _write_projection(
            scene_root / "response_projection.npz",
            method="response_projection",
            scene=scene,
            result=response,
            inputs=inputs,
            sources=sources,
            protocol_path=args.protocol,
            protocol_sha256=protocol_sha256,
            projection_seconds=response_seconds,
        )
        robust_metadata = _write_projection(
            scene_root / "robust_projection.npz",
            method="robust_projection",
            scene=scene,
            result=robust,
            inputs=inputs,
            sources=sources,
            protocol_path=args.protocol,
            protocol_sha256=protocol_sha256,
            projection_seconds=robust_seconds,
        )
        uniform_metadata = _write_projection(
            scene_root / "uniform_projection.npz",
            method="uniform_projection",
            scene=scene,
            result=uniform,
            inputs=inputs,
            sources=sources,
            protocol_path=args.protocol,
            protocol_sha256=protocol_sha256,
            projection_seconds=uniform_seconds,
        )
        order = {label: index for index, label in enumerate(labels)}
        native = max(
            sources,
            key=lambda record: (
                float(record["native_confidence_median"]),
                -order[str(record["label"])],
            ),
        )
        record = {
            "scene": scene,
            "response_projection": response_metadata["prediction"],
            "response_projection_sha256": response_metadata["prediction_sha256"],
            "robust_projection": robust_metadata["prediction"],
            "robust_projection_sha256": robust_metadata["prediction_sha256"],
            "uniform_projection": uniform_metadata["prediction"],
            "uniform_projection_sha256": uniform_metadata["prediction_sha256"],
            "orbit_medoid": robust["orbit_medoid"],
            "native_confidence": native["label"],
            "selected_response_basis": response["selected_basis"],
            "translation_status": response["translation_status"],
            "projection_seconds": (response_seconds + robust_seconds + uniform_seconds),
        }
        records.append(record)
        _checkpoint(args, protocol_sha256, records, complete=False)
        print(json.dumps({"event": "scene_complete", **record}), flush=True)
    _checkpoint(args, protocol_sha256, records, complete=True)
    print(
        json.dumps(
            {
                "status": "complete",
                "scene_count": len(records),
                "report": str(args.report.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
