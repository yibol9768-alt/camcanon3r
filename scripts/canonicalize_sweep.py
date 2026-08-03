#!/usr/bin/env python3
"""Prepare canonical-camera repair inputs for a multi-scene sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.protocol import list_images
from camcanon3r.repair import FILL_POLICIES, canonicalize_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--prefix", default="canonical_")
    parser.add_argument(
        "--fill-policy", choices=sorted(FILL_POLICIES), default="neutral_gray"
    )
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true")
    existing.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _output_name(variant: str, prefix: str) -> str:
    return "identity" if variant == "identity" else f"{prefix}{variant}"


def _is_complete(source: Path, output: Path) -> bool:
    if not output.is_dir() or not (output.parent / "manifest.json").is_file():
        return False
    source_names = {path.name for path in list_images(source)}
    output_names = {path.name for path in list_images(output)}
    mask_dir = output.parent / "_masks" / output.name
    mask_names = {path.name for path in mask_dir.glob("*.png")}
    return source_names == output_names == mask_names


def main() -> None:
    args = parse_args()
    executed: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for scene in args.scenes:
        for variant in args.variants:
            source = args.prepared_root / scene / variant
            output = args.output_root / scene / _output_name(variant, args.prefix)
            complete = _is_complete(source, output) if output.exists() else False
            if complete and args.resume:
                skipped.append({"scene": scene, "variant": variant})
                continue
            if output.exists() and not args.overwrite:
                state = "complete" if complete else "partial"
                raise FileExistsError(
                    f"{state} repair output requires --resume or --overwrite: "
                    f"{output}"
                )
            result = canonicalize_variant(
                source, output, fill_policy=args.fill_policy
            )
            valid_fractions = [
                item["repair"]["valid_fraction"] for item in result["images"]
            ]
            executed.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "output_variant": output.name,
                    "mean_valid_fraction": sum(valid_fractions)
                    / len(valid_fractions),
                }
            )
    print(
        json.dumps(
            {
                "executed_count": len(executed),
                "skipped_count": len(skipped),
                "executed": executed,
                "skipped": skipped,
            }
        )
    )


if __name__ == "__main__":
    main()
