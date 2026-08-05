#!/usr/bin/env python3
"""Re-audit a prepared canonical placement orbit without writing images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from camcanon3r.orbit_preparation import audit_canonical_orbit_scene
from camcanon3r.prediction import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_root", type=Path)
    parser.add_argument("orbit_root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.scenes or len(set(args.scenes)) != len(args.scenes):
        raise ValueError("scenes must be a non-empty unique list")
    records = []
    tree = hashlib.sha256()
    for scene in args.scenes:
        record = audit_canonical_orbit_scene(
            args.canonical_root / scene,
            args.orbit_root / scene,
            args.protocol,
        )
        records.append(record)
        tree.update(scene.encode("utf-8") + b"\0")
        tree.update(bytes.fromhex(str(record["tree_sha256"])))
    report = {
        "schema_version": "canonical-orbit-sweep-audit-0.1",
        "status": "complete",
        "canonical_root": str(args.canonical_root.resolve()),
        "orbit_root": str(args.orbit_root.resolve()),
        "protocol": str(args.protocol.resolve()),
        "scenes": args.scenes,
        "scene_count": len(records),
        "member_count": sum(int(record["member_count"]) for record in records),
        "image_count": sum(int(record["image_count"]) for record in records),
        "mask_count": sum(int(record["mask_count"]) for record in records),
        "decoded_rgb_matches": sum(
            int(record["decoded_rgb_matches"]) for record in records
        ),
        "decoded_mask_matches": sum(
            int(record["decoded_mask_matches"]) for record in records
        ),
        "tree_sha256": tree.hexdigest(),
        "records": records,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "scene_count", "image_count", "tree_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
