#!/usr/bin/env python3
"""Load VGGT once and run several prepared variants of one scene."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from run_vggt import load_model, run_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--preprocess", choices=("crop", "pad"), default="crop")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen VGGT pilot")
    device = torch.device("cuda")
    start = time.perf_counter()
    model = load_model(args.weights, device)
    load_seconds = time.perf_counter() - start
    summaries: list[dict[str, object]] = []
    for variant in args.variants:
        metadata = run_scene(
            scene_dir=args.prepared_root / variant,
            output=args.output_root / f"{variant}.npz",
            weights=args.weights,
            max_views=args.max_views,
            preprocess=args.preprocess,
            seed=args.seed,
            model=model,
            device=device,
            model_load_seconds=load_seconds,
            model_reused=True,
            print_metadata=False,
        )
        summaries.append(
            {
                "variant": variant,
                "inference_seconds": metadata["inference_seconds"],
                "peak_vram_bytes": metadata["peak_vram_bytes"],
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "model_load_seconds": load_seconds,
                "variant_count": len(summaries),
                "runs": summaries,
            }
        )
    )


if __name__ == "__main__":
    main()
