#!/usr/bin/env python3
"""Run official VGGT weights on one prepared scene without network access."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from camcanon3r.prediction import (
    PREDICTION_SCHEMA_VERSION,
    input_sha256_records,
    save_npz_compressed_atomic,
    write_json_atomic,
)
from camcanon3r.protocol import list_images, protocol_affines
from camcanon3r.vggt_preprocess import plan_vggt_preprocessing


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


def run_scene(
    *,
    scene_dir: Path,
    output: Path,
    weights: Path,
    max_views: int,
    preprocess: str,
    seed: int,
    model: VGGT,
    device: torch.device,
    model_load_seconds: float,
    model_reused: bool,
    print_metadata: bool = True,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    image_paths = list_images(scene_dir, max_views=max_views)
    source_sizes: list[tuple[int, int]] = []
    for path in image_paths:
        with Image.open(path) as opened:
            source_sizes.append(opened.size)
    preprocess_specs = plan_vggt_preprocessing(source_sizes, mode=preprocess)
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode=preprocess
    ).to(device)
    expected_size = preprocess_specs[0].affine.target_size
    if tuple(images.shape[-2:][::-1]) != expected_size:
        raise RuntimeError(
            f"logged preprocessing size {expected_size} does not match tensor "
            f"size {tuple(images.shape[-2:][::-1])}"
        )
    prepared_affines = protocol_affines(scene_dir, image_paths)
    model_affines = [spec.affine.matrix for spec in preprocess_specs]
    source_to_model = [
        model @ prepared
        for model, prepared in zip(model_affines, prepared_affines, strict=True)
    ]
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        predictions = model(images)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    save_npz_compressed_atomic(
        output,
        pose_enc=to_numpy(predictions["pose_enc"]),
        extrinsic=to_numpy(extrinsic),
        intrinsic=to_numpy(intrinsic),
        depth=to_numpy(predictions["depth"]),
        depth_conf=to_numpy(predictions["depth_conf"]),
        world_points=to_numpy(predictions["world_points"]),
        world_points_conf=to_numpy(predictions["world_points_conf"]),
        model_preprocess_affine=np.stack(model_affines),
        protocol_affine=np.stack(prepared_affines),
        source_to_model_affine=np.stack(source_to_model),
    )
    metadata = {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "scene_directory": str(scene_dir.resolve()),
        "inputs": [path.name for path in image_paths],
        "input_sha256": input_sha256_records(image_paths),
        "weights": str(weights.resolve()),
        "preprocess": preprocess,
        "seed": seed,
        "input_tensor_shape": list(images.shape),
        "spatial_transforms": [
            {
                "input": path.name,
                "input_size": list(spec.affine.source_size),
                "resized_size": list(spec.resized_size),
                "crop_top": spec.crop_top,
                "padding_left_top_right_bottom": list(spec.padding),
                "model_tensor_size": list(spec.affine.target_size),
                "model_preprocess_affine": spec.affine.matrix.tolist(),
                "protocol_affine": prepared.tolist(),
                "source_to_model_affine": combined.tolist(),
            }
            for path, spec, prepared, combined in zip(
                image_paths,
                preprocess_specs,
                prepared_affines,
                source_to_model,
                strict=True,
            )
        ],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "load_seconds": model_load_seconds,
        "model_reused_across_variants": model_reused,
        "inference_seconds": inference_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
    }
    metadata_path = output.with_suffix(".json")
    write_json_atomic(metadata_path, metadata)
    if print_metadata:
        print(json.dumps(metadata))
    return metadata


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen VGGT pilot")
    start = time.perf_counter()
    model = load_model(args.weights, device)
    load_seconds = time.perf_counter() - start
    run_scene(
        scene_dir=args.scene_dir,
        output=args.output,
        weights=args.weights,
        max_views=args.max_views,
        preprocess=args.preprocess,
        seed=args.seed,
        model=model,
        device=device,
        model_load_seconds=load_seconds,
        model_reused=False,
    )


if __name__ == "__main__":
    main()
