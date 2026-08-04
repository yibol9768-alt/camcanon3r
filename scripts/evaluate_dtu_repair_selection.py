#!/usr/bin/env python3
"""Evaluate the frozen DTU canonical-repair sweep and summarize atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from camcanon3r.dtu import evaluate_dtu_prediction
from camcanon3r.summary import summarize_dtu_evaluations

EVALUATION_PROTOCOL_VERSION = "dtu-repair-1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_root", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preparation-audit", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--point-variants", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_path(protocol_path: Path, reference: object) -> Path:
    path = Path(str(reference))
    if path.is_absolute():
        return path
    candidates = (
        Path.cwd() / path,
        protocol_path.resolve().parent.parent / path,
        protocol_path.resolve().parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"DTU repair protocol reference is missing: {path}")


def _load_design(
    protocol_path: Path,
    preparation_audit_path: Path,
    *,
    variants: list[str],
    point_variants: list[str],
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("frozen_before_dtu_gt_inspection"):
        raise ValueError("DTU repair protocol was not frozen before GT inspection")
    expected_variants = [str(value) for value in protocol["ordered_repaired_variants"]]
    expected_point_variants = [
        str(value) for value in protocol["point_metrics_variants"]
    ]
    if variants != expected_variants:
        raise ValueError(
            "DTU repair variants do not match the frozen ordered design: "
            f"expected={expected_variants}, actual={variants}"
        )
    if point_variants != expected_point_variants:
        raise ValueError(
            "DTU repair point variants do not match the frozen design: "
            f"expected={expected_point_variants}, actual={point_variants}"
        )
    if set(expected_point_variants) != set(expected_variants):
        raise ValueError(
            "frozen DTU repair design requires point metrics for all variants"
        )

    base_protocol_path = _reference_path(protocol_path, protocol["base_protocol"])
    base_protocol = json.loads(base_protocol_path.read_text(encoding="utf-8"))
    if not base_protocol.get("frozen_before_dtu_gt_inspection"):
        raise ValueError("base DTU protocol was not frozen before GT inspection")
    scans = [int(value) for value in base_protocol["evaluation_scans"]]
    camera_ids = [
        int(value) for value in base_protocol["rectified_archive_camera_ids_one_based"]
    ]
    if len(scans) != 22 or len(set(scans)) != len(scans):
        raise ValueError("DTU repair evaluation requires 22 unique frozen scans")
    if len(camera_ids) != 3 or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("DTU repair evaluation requires three unique camera IDs")
    scenes = [f"scan{scan}" for scan in scans]

    qualitative_path = _reference_path(protocol_path, protocol["qualitative_protocol"])
    qualitative = json.loads(qualitative_path.read_text(encoding="utf-8"))
    qualitative_variants = [str(value) for value in qualitative["variants"]]
    if not set(expected_variants).issubset(qualitative_variants):
        raise ValueError("DTU repair variants no longer satisfy qualitative protocol")

    audit = json.loads(preparation_audit_path.read_text(encoding="utf-8"))
    expected_audit = {
        "status": "complete",
        "scene_count": len(scenes),
        "scenes": scenes,
        "source_variants": [str(value) for value in protocol["source_variants"]],
        "fill_policy": protocol["fill_policy"],
        "output_variants": expected_variants,
        "image_count": len(scenes) * len(expected_variants) * len(camera_ids),
        "mask_count": len(scenes) * len(expected_variants) * len(camera_ids),
        "identity_pixel_matches": len(scenes) * len(camera_ids),
    }
    actual_audit = {key: audit.get(key) for key in expected_audit}
    if actual_audit != expected_audit:
        raise ValueError(
            "DTU repair preparation audit does not match the frozen design: "
            f"expected={expected_audit}, actual={actual_audit}"
        )
    tree_sha256 = audit.get("tree_sha256")
    if not isinstance(tree_sha256, str) or len(tree_sha256) != 64:
        raise ValueError("DTU repair preparation audit has no valid tree SHA-256")
    repaired_root = Path(str(audit["repaired_root"]))
    if not repaired_root.is_dir():
        raise FileNotFoundError(
            f"audited DTU repaired root is missing: {repaired_root}"
        )

    return {
        "protocol": protocol,
        "protocol_path": protocol_path.resolve(),
        "protocol_sha256": _sha256(protocol_path),
        "base_protocol_path": base_protocol_path,
        "base_protocol_sha256": _sha256(base_protocol_path),
        "qualitative_protocol_path": qualitative_path,
        "qualitative_protocol_sha256": _sha256(qualitative_path),
        "preparation_audit_path": preparation_audit_path.resolve(),
        "preparation_audit_sha256": _sha256(preparation_audit_path),
        "preparation_tree_sha256": tree_sha256,
        "repaired_root": repaired_root.resolve(),
        "scans": scans,
        "scenes": scenes,
        "camera_ids": camera_ids,
        "lighting": int(base_protocol["lighting_index"]),
        "variants": expected_variants,
        "point_variants": expected_point_variants,
    }


def _expected_jobs(
    selection_root: Path,
    prediction_root: Path,
    results_root: Path,
    protocol_path: Path,
    preparation_audit_path: Path,
    *,
    variants: list[str],
    point_variants: list[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    design = _load_design(
        protocol_path,
        preparation_audit_path,
        variants=variants,
        point_variants=point_variants,
    )
    calibration_dir = selection_root / "calibration/cal18"
    expected_names = [
        f"rect_{camera_id:03d}_{design['lighting']}_r5000.png"
        for camera_id in design["camera_ids"]
    ]
    for camera_id in design["camera_ids"]:
        path = calibration_dir / f"pos_{camera_id:03d}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"DTU calibration is missing: {path}")

    jobs: list[dict[str, object]] = []
    point_variant_set = set(point_variants)
    for scan, scene in zip(design["scans"], design["scenes"], strict=True):
        source_dir = selection_root / "source" / scene
        for name in expected_names:
            path = source_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"frozen DTU source image is missing: {path}")
        required_gt = (
            selection_root / f"gt/Points/stl/stl{scan:03d}_total.ply",
            selection_root / f"gt/ObsMask/ObsMask{scan}_10.mat",
            selection_root / f"gt/ObsMask/Plane{scan}.mat",
        )
        missing_gt = [str(path) for path in required_gt if not path.is_file()]
        if missing_gt:
            raise FileNotFoundError(
                f"frozen DTU point GT is missing for {scene}: {missing_gt}"
            )
        for variant in variants:
            prediction = prediction_root / scene / f"{variant}.npz"
            metadata_path = prediction.with_suffix(".json")
            if not prediction.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    f"complete DTU repair prediction is missing: "
                    f"{prediction} and {metadata_path}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("inputs") != expected_names:
                raise RuntimeError(
                    f"repair prediction inputs do not match frozen DTU views for "
                    f"{scene}/{variant}: expected={expected_names}, "
                    f"actual={metadata.get('inputs')}"
                )
            expected_scene_dir = str(
                (Path(design["repaired_root"]) / scene / variant).resolve()
            )
            if metadata.get("scene_directory") != expected_scene_dir:
                raise RuntimeError(
                    f"repair prediction is not bound to the audited input tree: "
                    f"{scene}/{variant}"
                )
            point_requested = variant in point_variant_set
            jobs.append(
                {
                    "scene": scene,
                    "scan": scan,
                    "variant": variant,
                    "prediction": prediction,
                    "calibration_dir": calibration_dir,
                    "gt_root": selection_root / "gt" if point_requested else None,
                    "point_metrics_requested": point_requested,
                    "output": results_root / scene / f"{variant}_vs_gt.json",
                    "protocol_sha256": design["protocol_sha256"],
                    "base_protocol_sha256": design["base_protocol_sha256"],
                    "qualitative_protocol_sha256": design[
                        "qualitative_protocol_sha256"
                    ],
                    "preparation_audit_sha256": design["preparation_audit_sha256"],
                    "preparation_tree_sha256": design["preparation_tree_sha256"],
                }
            )
    return jobs, design


def _validate_existing(record: dict[str, object], job: dict[str, object]) -> None:
    fields = (
        "scene",
        "scan",
        "variant",
        "point_metrics_requested",
        "protocol_sha256",
        "base_protocol_sha256",
        "qualitative_protocol_sha256",
        "preparation_audit_sha256",
        "preparation_tree_sha256",
    )
    expected = {field: job[field] for field in fields}
    expected.update(
        {
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "prediction": str(Path(job["prediction"]).resolve()),
            "calibration_dir": str(Path(job["calibration_dir"]).resolve()),
            "gt_root": (
                str(Path(job["gt_root"]).resolve()) if job["gt_root"] else None
            ),
        }
    )
    actual = {key: record.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "existing DTU repair evaluation does not match frozen job: "
            f"expected={expected}, actual={actual}"
        )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    jobs, design = _expected_jobs(
        args.selection_root,
        args.prediction_root,
        args.results_root,
        args.protocol,
        args.preparation_audit,
        variants=args.variants,
        point_variants=args.point_variants,
    )
    expected_outputs = {Path(job["output"]).resolve() for job in jobs}
    unexpected_outputs = {
        path.resolve() for path in args.results_root.rglob("*_vs_gt.json")
    } - expected_outputs
    if unexpected_outputs:
        raise RuntimeError(
            "results root contains evaluations outside the frozen DTU repair design: "
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
        result = evaluate_dtu_prediction(
            Path(job["prediction"]),
            Path(job["calibration_dir"]),
            scan=int(job["scan"]),
            gt_root=(Path(job["gt_root"]) if job["gt_root"] else None),
        )
        result.update(
            {
                "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
                "scene": job["scene"],
                "variant": job["variant"],
                "calibration_dir": str(Path(job["calibration_dir"]).resolve()),
                "gt_root": str(Path(job["gt_root"]).resolve()),
                "point_metrics_requested": job["point_metrics_requested"],
                "protocol_sha256": job["protocol_sha256"],
                "base_protocol_sha256": job["base_protocol_sha256"],
                "qualitative_protocol_sha256": job["qualitative_protocol_sha256"],
                "preparation_audit_sha256": job["preparation_audit_sha256"],
                "preparation_tree_sha256": job["preparation_tree_sha256"],
            }
        )
        _write_json_atomic(output, result)
        completed += 1
        print(
            json.dumps(
                {
                    "scene": job["scene"],
                    "variant": job["variant"],
                    "point_metrics": job["point_metrics_requested"],
                    "status": "complete",
                }
            ),
            flush=True,
        )

    outputs = sorted(Path(job["output"]) for job in jobs)
    summary = summarize_dtu_evaluations(
        outputs,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "protocol": str(design["protocol_path"]),
            "protocol_sha256": design["protocol_sha256"],
            "base_protocol": str(design["base_protocol_path"]),
            "base_protocol_sha256": design["base_protocol_sha256"],
            "qualitative_protocol": str(design["qualitative_protocol_path"]),
            "qualitative_protocol_sha256": design["qualitative_protocol_sha256"],
            "preparation_audit": str(design["preparation_audit_path"]),
            "preparation_audit_sha256": design["preparation_audit_sha256"],
            "preparation_tree_sha256": design["preparation_tree_sha256"],
            "prediction_root": str(args.prediction_root.resolve()),
            "variants": args.variants,
            "point_variants": args.point_variants,
        }
    )
    summary_output = args.summary_output or args.results_root / "summary.json"
    _write_json_atomic(summary_output, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "evaluation_count": len(jobs),
                "completed_count": completed,
                "skipped_count": skipped,
                "summary": str(summary_output),
            }
        )
    )


if __name__ == "__main__":
    main()
