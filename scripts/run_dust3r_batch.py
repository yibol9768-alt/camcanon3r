#!/usr/bin/env python3
"""Load DUSt3R once and run a resumable multi-scene sweep."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from run_dust3r import load_model, run_scene

from camcanon3r.sweep import plan_prediction_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=(224, 512), default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true")
    existing.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planned = plan_prediction_sweep(
        args.prepared_root,
        args.output_root,
        variants=args.variants,
        scenes=args.scenes,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    pending = [run for run in planned if not run.skip]
    if not pending:
        print(
            json.dumps(
                {
                    "model_load_seconds": None,
                    "run_count": len(planned),
                    "executed_count": 0,
                    "skipped_count": len(planned),
                }
            )
        )
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen DUSt3R protocol")
    device = torch.device("cuda")
    start = time.perf_counter()
    model = load_model(args.weights, device)
    load_seconds = time.perf_counter() - start
    summaries: list[dict[str, object]] = []
    for run in planned:
        if run.skip:
            summaries.append(
                {"scene": run.scene, "variant": run.variant, "status": "skipped"}
            )
            continue
        metadata = run_scene(
            scene_dir=run.prepared_dir,
            output=run.output,
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
            model_reused=True,
            print_metadata=False,
        )
        summaries.append(
            {
                "scene": run.scene,
                "variant": run.variant,
                "status": "executed",
                "pairwise_inference_seconds": metadata[
                    "pairwise_inference_seconds"
                ],
                "alignment_seconds": metadata["alignment_seconds"],
                "peak_vram_bytes": metadata["peak_vram_bytes"],
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "model_load_seconds": load_seconds,
                "run_count": len(summaries),
                "executed_count": len(pending),
                "skipped_count": len(planned) - len(pending),
                "runs": summaries,
            }
        )
    )


if __name__ == "__main__":
    main()
