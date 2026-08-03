#!/usr/bin/env python3
"""Run official VGGT weights on one prepared scene without network access."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from camcanon3r.protocol import list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--preprocess", choices=("crop", "pad"), default="crop")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def load_model(weights: Path, device: torch.device) -> VGGT:
    model = VGGT(enable_track=False)
    state = load_file(str(weights), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"missing VGGT weights: {missing[:8]}")
    invalid_unexpected = [
        key for key in unexpected if not key.startswith("track_head.")
    ]
    if invalid_unexpected:
        raise RuntimeError(f"unexpected non-track weights: {invalid_unexpected[:8]}")
    return model.eval().to(device)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen VGGT pilot")

    image_paths = list_images(args.scene_dir, max_views=args.max_views)
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode=args.preprocess
    ).to(device)
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = load_model(args.weights, device)
    load_seconds = time.perf_counter() - start

    inference_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        predictions = model(images)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pose_enc=to_numpy(predictions["pose_enc"]),
        extrinsic=to_numpy(extrinsic),
        intrinsic=to_numpy(intrinsic),
        depth=to_numpy(predictions["depth"]),
        depth_conf=to_numpy(predictions["depth_conf"]),
        world_points=to_numpy(predictions["world_points"]),
        world_points_conf=to_numpy(predictions["world_points_conf"]),
    )
    metadata = {
        "scene_directory": str(args.scene_dir.resolve()),
        "inputs": [path.name for path in image_paths],
        "weights": str(args.weights.resolve()),
        "preprocess": args.preprocess,
        "seed": args.seed,
        "input_tensor_shape": list(images.shape),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
