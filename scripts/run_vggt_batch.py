#!/usr/bin/env python3
"""Load VGGT once and run a resumable multi-scene variant sweep."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from run_vggt import load_model, run_scene

from camcanon3r.sweep import plan_prediction_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="scene directories beneath prepared_root; omit for one scene root",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--preprocess", choices=("crop", "pad"), default="crop")
    parser.add_argument("--seed", type=int, default=17)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        "--resume",
        action="store_true",
        help="skip variants with both NPZ and JSON outputs already present",
    )
    existing.add_argument(
        "--overwrite",
        action="store_true",
        help="replace complete or partial outputs",
    )
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
                    "runs": [
                        {
                            "scene": run.scene,
                            "variant": run.variant,
                            "status": "skipped",
                        }
                        for run in planned
                    ],
                }
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen VGGT pilot")
    device = torch.device("cuda")
    start = time.perf_counter()
    model = load_model(args.weights, device)
    load_seconds = time.perf_counter() - start
    summaries: list[dict[str, object]] = []
    for run in planned:
        if run.skip:
            event = {
                "scene": run.scene,
                "variant": run.variant,
                "status": "skipped",
            }
            summaries.append(event)
            print(json.dumps({"event": "run_complete", **event}), flush=True)
            continue
        metadata = run_scene(
            scene_dir=run.prepared_dir,
            output=run.output,
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
        event = {
            "scene": run.scene,
            "variant": run.variant,
            "status": "executed",
            "inference_seconds": metadata["inference_seconds"],
            "peak_vram_bytes": metadata["peak_vram_bytes"],
        }
        summaries.append(event)
        print(json.dumps({"event": "run_complete", **event}), flush=True)
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
