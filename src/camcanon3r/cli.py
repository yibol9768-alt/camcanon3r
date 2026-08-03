"""Command-line entry points for CamCanon3R."""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import prepare_scene

DEFAULT_VARIANTS = (
    "identity",
    "center_crop_075",
    "asymmetric_crop_075",
    "letterbox_square",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="camcanon3r")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-scene")
    prepare.add_argument("scene_dir", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--max-views", type=int)
    prepare.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-scene":
        manifest = prepare_scene(
            args.scene_dir,
            args.output_dir,
            variants=args.variants,
            seed=args.seed,
            max_views=args.max_views,
        )
        print(
            f"prepared {len(manifest['variants'])} variants in {args.output_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
