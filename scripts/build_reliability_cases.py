#!/usr/bin/env python3
"""Build complete reliability cases from predictions and GT evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camcanon3r.reliability_cases import build_reliability_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_reliability_cases(
        args.prediction_root,
        args.result_root,
        variants=tuple(args.variants),
        model=args.model,
        dataset=args.dataset,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "scene_count": result["scene_count"],
                "case_count": result["case_count"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
