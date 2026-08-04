#!/usr/bin/env python3
"""Prepare every scene in a frozen extracted ETH3D selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.protocol import list_images, prepare_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_root", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("--domain", choices=("raw", "undistorted"), required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.selection_root / "selection_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("completed_at") is None:
        raise RuntimeError("ETH3D selection report is not complete")
    source_subdirectory = (
        Path("images/dslr_images")
        if args.domain == "raw"
        else Path("images/dslr_images_undistorted")
    )
    scene_records = report["selection"]["scenes"]
    completed: list[str] = []
    for scene_record in scene_records:
        scene = str(scene_record["scene"])
        expected_names = list(scene_record["image_names"])
        source_dir = args.selection_root / scene / source_subdirectory
        actual_names = [path.name for path in list_images(source_dir)]
        if actual_names != expected_names:
            raise RuntimeError(
                f"selected inputs do not match report for {scene}: "
                f"expected={expected_names}, actual={actual_names}"
            )
        output_dir = args.prepared_root / scene
        if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
            raise FileExistsError(
                f"prepared scene exists; use --resume: {output_dir}"
            )
        manifest = prepare_scene(
            source_dir,
            output_dir,
            variants=args.variants,
            seed=args.seed,
            scene_name=scene,
            max_views=len(expected_names),
            resume=args.resume,
        )
        completed.append(scene)
        print(
            json.dumps(
                {
                    "scene": scene,
                    "domain": args.domain,
                    "variant_count": len(manifest["variants"]),
                    "status": "complete",
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "domain": args.domain,
                "scene_count": len(completed),
                "scenes": completed,
            }
        )
    )


if __name__ == "__main__":
    main()
