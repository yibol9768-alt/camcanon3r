#!/usr/bin/env python3
"""Run pinned official DUSt3R weights on one prepared scene."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from camcanon3r.dust3r_preprocess import plan_dust3r_preprocessing
from camcanon3r.prediction import (
    stack_equal_shapes,
    world_to_camera_from_camera_to_world,
)
from camcanon3r.protocol import list_images, protocol_affines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="official .pth file or local Hugging Face snapshot directory",
    )
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def load_model(weights: Path, device: Any) -> Any:
    from dust3r.model import AsymmetricCroCo3DStereo

    if not weights.exists():
        raise FileNotFoundError(f"DUSt3R checkpoint is missing: {weights}")
    if weights.is_dir():
        required = [weights / "config.json", weights / "model.safetensors"]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"DUSt3R snapshot is incomplete at {weights}: missing {missing}"
            )
    return AsymmetricCroCo3DStereo.from_pretrained(str(weights)).eval().to(device)


def _source_sizes(image_paths: list[Path]) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for path in image_paths:
        with Image.open(path) as opened:
            sizes.append(ImageOps.exif_transpose(opened).size)
    return sizes


def run_scene(
    *,
    scene_dir: Path,
    output: Path,
    weights: Path,
    max_views: int,
    image_size: int,
    batch_size: int,
    niter: int,
    schedule: str,
    lr: float,
    seed: int,
    model: Any,
    device: Any,
    model_load_seconds: float,
    model_reused: bool,
    print_metadata: bool = True,
) -> dict[str, object]:
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    if max_views < 3:
        raise ValueError("confirmatory DUSt3R runs require at least three views")
    if niter <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("batch size, niter, and learning rate must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    image_paths = list_images(scene_dir, max_views=max_views)
    source_sizes = _source_sizes(image_paths)
    patch_size = int(model.patch_size)
    square_ok = bool(getattr(model, "square_ok", False))
    preprocess_specs = plan_dust3r_preprocessing(
        source_sizes,
        image_size=image_size,
        patch_size=patch_size,
        square_ok=square_ok,
    )
    images = load_images(
        [str(path) for path in image_paths],
        size=image_size,
        patch_size=patch_size,
        square_ok=square_ok,
        verbose=False,
    )
    if len(images) != len(preprocess_specs):
        raise RuntimeError("DUSt3R loader did not return one tensor per input")
    for image, spec in zip(images, preprocess_specs, strict=True):
        true_height, true_width = np.asarray(image["true_shape"]).reshape(-1, 2)[0]
        actual_size = (int(true_width), int(true_height))
        if actual_size != spec.affine.target_size:
            raise RuntimeError(
                f"logged DUSt3R size {spec.affine.target_size} does not match "
                f"tensor size {actual_size}"
            )

    prepared_affines = protocol_affines(scene_dir, image_paths)
    model_affines = [spec.affine.matrix for spec in preprocess_specs]
    source_to_model = [
        model_affine @ prepared_affine
        for model_affine, prepared_affine in zip(
            model_affines, prepared_affines, strict=True
        )
    ]
    pairs = make_pairs(
        images, scene_graph="complete", prefilter=None, symmetrize=True
    )
    torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()
    pairwise = inference(
        pairs, model, device, batch_size=batch_size, verbose=False
    )
    torch.cuda.synchronize(device)
    pairwise_seconds = time.perf_counter() - inference_start

    align_start = time.perf_counter()
    scene = global_aligner(
        pairwise,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False,
    )
    alignment_loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule=schedule, lr=lr
    )
    torch.cuda.synchronize(device)
    alignment_seconds = time.perf_counter() - align_start

    cam2world = to_numpy(scene.get_im_poses())
    extrinsic = world_to_camera_from_camera_to_world(cam2world)
    intrinsic = to_numpy(scene.get_intrinsics())
    depth = stack_equal_shapes(
        (to_numpy(item) for item in scene.get_depthmaps()), label="depth"
    )
    points = stack_equal_shapes(
        (to_numpy(item) for item in scene.get_pts3d()), label="world points"
    )
    confidence = stack_equal_shapes(
        (to_numpy(item) for item in scene.im_conf), label="confidence"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        cam2world=cam2world,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth=depth,
        depth_conf=confidence,
        world_points=points,
        world_points_conf=confidence,
        model_preprocess_affine=np.stack(model_affines),
        protocol_affine=np.stack(prepared_affines),
        source_to_model_affine=np.stack(source_to_model),
    )
    metadata: dict[str, object] = {
        "model": "DUSt3R",
        "scene_directory": str(scene_dir.resolve()),
        "inputs": [path.name for path in image_paths],
        "weights": str(weights.resolve()),
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "scene_graph": "complete_symmetrized",
        "alignment": {
            "mode": "PointCloudOptimizer",
            "init": "mst",
            "niter": niter,
            "schedule": schedule,
            "lr": lr,
            "loss": float(alignment_loss),
        },
        "spatial_transforms": [
            {
                "input": path.name,
                "input_size": list(spec.affine.source_size),
                "resized_size": list(spec.resized_size),
                "crop_left_top_right_bottom": list(
                    spec.crop_left_top_right_bottom
                ),
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
        "load_seconds": model_load_seconds,
        "model_reused_across_variants": model_reused,
        "pairwise_inference_seconds": pairwise_seconds,
        "alignment_seconds": alignment_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if print_metadata:
        print(json.dumps(metadata))
    return metadata


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen DUSt3R protocol")
    device = torch.device("cuda")
    start = time.perf_counter()
    model = load_model(args.weights, device)
    load_seconds = time.perf_counter() - start
    run_scene(
        scene_dir=args.scene_dir,
        output=args.output,
        weights=args.weights,
        max_views=args.max_views,
        image_size=args.image_size,
        batch_size=args.batch_size,
        niter=args.niter,
        schedule=args.schedule,
        lr=args.lr,
        seed=args.seed,
        model=model,
        device=device,
        model_load_seconds=load_seconds,
        model_reused=False,
    )


if __name__ == "__main__":
    main()
