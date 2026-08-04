#!/usr/bin/env python3
"""Render the registered crop-severity result directly from an analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = {
    "Center": "center_crop_{fraction}",
    "Shared off-center": "shared_asymmetric_crop_{fraction}",
    "Independent off-center": "asymmetric_crop_{fraction}",
}
FRACTIONS = (90, 75, 60)
METRIC = "rotation_median_degrees_delta_from_identity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--png", type=Path)
    return parser.parse_args()


def _metric(record: dict[str, object]) -> tuple[float, float, float]:
    interval = record["metrics"][METRIC]
    return (
        float(interval["point_estimate"]),
        float(interval["lower"]),
        float(interval["upper"]),
    )


def _analyses(payload: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    for key, record in payload["analyses"].items():
        model = str(record["model"])
        dataset = str(record["dataset"])
        if key != f"{model}/{dataset}":
            raise ValueError(f"analysis key/provenance mismatch: {key}")
        result[(model, dataset)] = record["analysis"]
    if not result:
        raise ValueError("mechanism analysis contains no model/dataset records")
    return result


def main() -> None:
    args = parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    analyses = _analyses(payload)

    import matplotlib.pyplot as plt

    models = [
        value
        for value in ("vggt", "dust3r")
        if any(key[0] == value for key in analyses)
    ]
    models += sorted({key[0] for key in analyses} - set(models))
    datasets = [
        value for value in ("eth3d", "dtu") if any(key[1] == value for key in analyses)
    ]
    datasets += sorted({key[1] for key in analyses} - set(datasets))
    figure, axes = plt.subplots(
        len(models),
        len(datasets),
        figsize=(3.25 * len(datasets), 2.05 * len(models)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = ("#4C78A8", "#F58518", "#B22222")
    markers = ("o", "s", "^")
    for row, model in enumerate(models):
        for column, dataset in enumerate(datasets):
            axis = axes[row][column]
            analysis = analyses.get((model, dataset))
            if analysis is None:
                axis.set_axis_off()
                continue
            by_variant = analysis["by_variant"]
            for (label, pattern), color, marker in zip(
                FAMILIES.items(), colors, markers, strict=True
            ):
                estimates: list[float] = []
                lowers: list[float] = []
                uppers: list[float] = []
                for fraction in FRACTIONS:
                    variant = pattern.format(fraction=f"{fraction:03d}")
                    estimate, lower, upper = _metric(by_variant[variant])
                    estimates.append(estimate)
                    lowers.append(estimate - lower)
                    uppers.append(upper - estimate)
                axis.errorbar(
                    FRACTIONS,
                    estimates,
                    yerr=[lowers, uppers],
                    color=color,
                    marker=marker,
                    linewidth=1.4,
                    markersize=4.2,
                    capsize=2.0,
                    label=label,
                )
            axis.axhline(2.0, color="#666666", linestyle="--", linewidth=0.9)
            axis.set_xlim(93, 57)
            axis.set_xticks(FRACTIONS)
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
            axis.set_title(f"{model.upper()} — {dataset.upper()}", fontsize=8)
            if row == len(models) - 1:
                axis.set_xlabel("Retained fraction (%)", fontsize=8)
            axis.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.supylabel("Paired rotation increase (°)", fontsize=8)
    figure.tight_layout(rect=(0.035, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.png, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
