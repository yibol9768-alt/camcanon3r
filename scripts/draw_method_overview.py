#!/usr/bin/env python3
"""Draw the paper's method overview as a deterministic vector figure."""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--png", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(7.0, 2.35))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        *,
        face: str,
        edge: str,
        size: float = 7.5,
        style: str = "round,pad=0.012,rounding_size=0.012",
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=style,
                linewidth=1.0,
                facecolor=face,
                edgecolor=edge,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            linespacing=1.25,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = "#444444",
        dashed: bool = False,
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.05,
                linestyle="--" if dashed else "-",
                color=color,
                shrinkA=1,
                shrinkB=1,
            )
        )

    blue_face = "#EAF2F8"
    blue_edge = "#4C78A8"
    amber_face = "#FFF3E0"
    amber_edge = "#F58518"
    gray_face = "#F2F2F2"
    gray_edge = "#666666"
    red_face = "#FCEBEC"
    red_edge = "#B22222"
    green_face = "#EDF7ED"
    green_edge = "#4C8C4A"

    top_y, top_h, top_w = 0.62, 0.25, 0.18
    xs = (0.02, 0.275, 0.53, 0.785)
    box(
        xs[0],
        top_y,
        top_w,
        top_h,
        "Source views\n$ I_i,\\; K_i $",
        face=blue_face,
        edge=blue_edge,
    )
    box(
        xs[1],
        top_y,
        top_w,
        top_h,
        "Logged affine $A_i$\ncrop / resize / pad",
        face=amber_face,
        edge=amber_edge,
    )
    box(
        xs[2],
        top_y,
        top_w,
        top_h,
        "Model preprocessing\n+ frozen network\n$C_i=B_iA_i$",
        face=gray_face,
        edge=gray_edge,
    )
    box(
        xs[3],
        top_y,
        top_w,
        top_h,
        "Predicted geometry\n$\\widehat K_i,\\;\\widehat E_i,\\;\\widehat X_i$",
        face=red_face,
        edge=red_edge,
    )
    for first, second in pairwise(xs):
        arrow((first + top_w, top_y + top_h / 2), (second, top_y + top_h / 2))

    box(
        0.37,
        0.25,
        0.38,
        0.22,
        "Common-domain audit\nmap pixels/intrinsics by $C_i^{-1}$\nremove only global Sim(3)",
        face=blue_face,
        edge=blue_edge,
        size=7.2,
    )
    box(
        0.79,
        0.25,
        0.18,
        0.22,
        "Paired evidence\nidentity gap, GT error,\ndisagreement / risk",
        face=red_face,
        edge=red_edge,
        size=7.0,
    )
    arrow((xs[3] + top_w / 2, top_y), (0.68, 0.47), color=red_edge)
    arrow((0.75, 0.36), (0.79, 0.36), color=red_edge)

    box(
        0.02,
        0.16,
        0.27,
        0.25,
        "Analytic canonicalization\n$A_i^{-1}$ warp + validity mask\nfixed fill (lost pixels stay lost)",
        face=green_face,
        edge=green_edge,
        size=7.2,
    )
    arrow((xs[1] + top_w / 2, top_y), (0.155, 0.41), color=green_edge, dashed=True)
    arrow((0.29, 0.285), (0.37, 0.285), color=green_edge, dashed=True)
    axis.text(
        0.155,
        0.08,
        "Repair claim is promoted only when paired GT improves; lost pixels stay lost.",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#2F6B2F",
    )
    axis.text(
        0.02,
        0.95,
        "Known camera intervention",
        fontsize=7.2,
        weight="bold",
        color=amber_edge,
    )
    axis.text(
        0.53,
        0.95,
        "Frozen model response",
        fontsize=7.2,
        weight="bold",
        color=gray_edge,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.03)
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.png, dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


if __name__ == "__main__":
    main()
