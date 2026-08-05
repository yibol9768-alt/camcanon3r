#!/usr/bin/env python3
"""Prepare and audit a frozen canonical-placement orbit across scenes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

from camcanon3r.orbit_preparation import (
    audit_canonical_orbit_scene,
    prepare_canonical_orbit_scene,
)
from camcanon3r.prediction import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _design(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": "canonical-orbit-sweep-0.1",
        "canonical_root": str(args.canonical_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "protocol": str(args.protocol.resolve()),
        "scenes": [str(scene) for scene in args.scenes],
    }


def _load_records(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.report.exists():
        return []
    if not args.resume:
        raise FileExistsError(f"orbit report exists; use --resume: {args.report}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    design = _design(args)
    if {key: report.get(key) for key in design} != design:
        raise ValueError("existing orbit preparation report design does not match")
    records = report.get("records")
    if not isinstance(records, list):
        raise TypeError("existing orbit preparation records are invalid")
    return [dict(record) for record in records]


def _checkpoint(
    args: argparse.Namespace,
    records: list[dict[str, object]],
    *,
    complete: bool,
) -> None:
    durations = [float(record["preparation_seconds"]) for record in records]
    write_json_atomic(
        args.report,
        {
            **_design(args),
            "status": "complete" if complete else "in_progress",
            "record_count": len(records),
            "preparation_seconds": {
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
    records = _load_records(args)
    completed = {str(record["scene"]): record for record in records}
    if set(completed) - set(args.scenes):
        raise ValueError("existing orbit report contains an unexpected scene")
    for scene in args.scenes:
        canonical_scene = args.canonical_root / scene
        output_scene = args.output_root / scene
        if scene in completed:
            audit = audit_canonical_orbit_scene(
                canonical_scene, output_scene, args.protocol
            )
            if audit.get("tree_sha256") != completed[scene].get("tree_sha256"):
                raise ValueError(f"resumed orbit tree hash changed: {scene}")
            continue
        start = time.perf_counter()
        prepare_canonical_orbit_scene(
            canonical_scene,
            output_scene,
            args.protocol,
            resume=args.resume,
        )
        duration = time.perf_counter() - start
        audit = audit_canonical_orbit_scene(
            canonical_scene, output_scene, args.protocol
        )
        record = {
            "scene": scene,
            "preparation_seconds": duration,
            "tree_sha256": audit["tree_sha256"],
            "member_count": audit["member_count"],
            "image_count": audit["image_count"],
            "mask_count": audit["mask_count"],
            "decoded_rgb_matches": audit["decoded_rgb_matches"],
            "decoded_mask_matches": audit["decoded_mask_matches"],
        }
        records.append(record)
        completed[scene] = record
        _checkpoint(args, records, complete=False)
        print(json.dumps({"event": "scene_complete", **record}), flush=True)
    ordered = [completed[scene] for scene in args.scenes]
    _checkpoint(args, ordered, complete=True)
    print(
        json.dumps(
            {
                "status": "complete",
                "scene_count": len(ordered),
                "report": str(args.report.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
