#!/usr/bin/env python3
"""Render score and oracle risk--coverage curves from frozen reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from camcanon3r.reliability import resolve_case_field, risk_coverage_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--panel",
        action="append",
        nargs=4,
        metavar=("LABEL", "DISAGREEMENT_REPORT", "NATIVE_REPORT", "CASES"),
        required=True,
    )
    parser.add_argument("--png", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cases(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise TypeError(f"reliability cases are not a non-empty list: {path}")
    return records


def _step(axis: object, curve: dict[str, object], **kwargs: object) -> None:
    axis.step(
        [0.0, *curve["coverage"]],
        [curve["risk"][0], *curve["risk"]],
        where="post",
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    panels: list[dict[str, object]] = []
    for label, disagreement_name, native_name, cases_name in args.panel:
        disagreement_path = Path(disagreement_name)
        native_path = Path(native_name)
        cases_path = Path(cases_name)
        disagreement = json.loads(disagreement_path.read_text(encoding="utf-8"))
        native = json.loads(native_path.read_text(encoding="utf-8"))
        cases_sha256 = _sha256(cases_path)
        for report in (disagreement, native):
            if report["input"]["cases_sha256"] != cases_sha256:
                raise ValueError(f"report/cases SHA-256 mismatch for panel {label}")
        error_field = disagreement["input"]["error_field"]
        if native["input"]["error_field"] != error_field:
            raise ValueError(f"report error fields differ for panel {label}")
        records = _load_cases(cases_path)
        errors = [float(resolve_case_field(record, error_field)) for record in records]
        oracle = disagreement.get("oracle_risk_coverage")
        if oracle is None:
            oracle = risk_coverage_curve(errors, errors)
        panels.append(
            {
                "label": label,
                "disagreement": disagreement,
                "native": native,
                "oracle": oracle,
            }
        )

    import matplotlib.pyplot as plt

    columns = min(2, len(panels))
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.25 * columns, 2.2 * rows),
        sharex=True,
        squeeze=False,
    )
    for index, panel in enumerate(panels):
        axis = axes[index // columns][index % columns]
        disagreement = panel["disagreement"]
        native = panel["native"]
        oracle = panel["oracle"]
        _step(
            axis,
            disagreement["risk_coverage"],
            color="#B22222",
            linewidth=1.5,
            label=f"Disagreement ({disagreement['aurc']['point_estimate']:.2f})",
        )
        _step(
            axis,
            native["risk_coverage"],
            color="#4C78A8",
            linewidth=1.3,
            label=f"Native ({native['aurc']['point_estimate']:.2f})",
        )
        _step(
            axis,
            oracle,
            color="#333333",
            linewidth=1.1,
            linestyle="--",
            label=f"Oracle ({disagreement['oracle_aurc']['point_estimate']:.2f})",
        )
        axis.set_xlim(0.0, 1.0)
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.set_title(str(panel["label"]), fontsize=8)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Mean rotation error (°)")
        axis.legend(frameon=False, fontsize=6.5)
    for index in range(len(panels), rows * columns):
        axes[index // columns][index % columns].set_axis_off()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.png, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
