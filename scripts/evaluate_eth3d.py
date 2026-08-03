#!/usr/bin/env python3
"""Evaluate one VGGT result against official ETH3D poses and depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from camcanon3r.eth3d import open_eth3d_depth, read_colmap_cameras, read_colmap_images
from camcanon3r.metrics import (
    aligned_depth_to_source_ground_truth,
    pairwise_relative_pose_errors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("calibration_dir", type=Path)
    parser.add_argument("depth_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    return {
        "count": len(finite),
        "median": float(np.median(finite)) if len(finite) else None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.prediction.with_suffix(".json").read_text())
    cameras = read_colmap_cameras(args.calibration_dir / "cameras.txt")
    images = read_colmap_images(args.calibration_dir / "images.txt")
    inputs = metadata["inputs"]
    matched = [images[Path(name).stem] for name in inputs]
    ground_truth_extrinsics = np.stack([item.extrinsic for item in matched])
    camera = cameras[matched[0].camera_id]
    if any(item.camera_id != camera.camera_id for item in matched):
        raise RuntimeError("the selected ETH3D views do not share one camera model")

    with np.load(args.prediction) as prediction:
        pose_errors = pairwise_relative_pose_errors(
            ground_truth_extrinsics, prediction["extrinsic"]
        )
        source_depths = [
            open_eth3d_depth(
                args.depth_dir / f"{Path(name).stem}.JPG",
                width=camera.width,
                height=camera.height,
            )
            for name in inputs
        ]
        depth = aligned_depth_to_source_ground_truth(
            prediction["depth"],
            prediction["source_to_model_affine"],
            source_depths,
        )

    result = {
        "prediction": str(args.prediction.resolve()),
        "inputs": inputs,
        "camera_model": camera.model,
        "camera_size": [camera.width, camera.height],
        "relative_rotation_degrees": finite_summary(pose_errors["rotation_degrees"]),
        "translation_direction_degrees": finite_summary(
            pose_errors["translation_direction_degrees"]
        ),
        "depth": depth,
    }
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
