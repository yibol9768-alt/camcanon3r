"""Aggregate scene-level CamCanon3R comparison records."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize_comparison_files(
    paths: list[Path],
    *,
    rotation_threshold: float = 2.0,
    depth_threshold: float = 0.05,
) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one comparison record is required")
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        reference = Path(record["reference"])
        row = {
            "scene": reference.parent.name,
            "candidate": record["candidate_label"],
            "rotation_median_degrees": record["rotation_degrees"]["median"],
            "translation_median_degrees": record[
                "translation_direction_degrees"
            ]["median"],
            "depth_mean_abs_rel": record["aligned_depth_consistency"][
                "mean_abs_rel"
            ],
            "valid_depth_pixels": record["aligned_depth_consistency"][
                "valid_pixels"
            ],
            "source": str(path),
        }
        rows.append(row)
        grouped[str(row["candidate"])].append(row)

    by_variant: dict[str, dict[str, object]] = {}
    for candidate, candidate_rows in sorted(grouped.items()):
        rotations = np.asarray(
            [row["rotation_median_degrees"] for row in candidate_rows],
            dtype=np.float64,
        )
        translations = np.asarray(
            [row["translation_median_degrees"] for row in candidate_rows],
            dtype=np.float64,
        )
        depths = np.asarray(
            [row["depth_mean_abs_rel"] for row in candidate_rows], dtype=np.float64
        )
        by_variant[candidate] = {
            "scene_count": len(candidate_rows),
            "scenes": sorted(str(row["scene"]) for row in candidate_rows),
            "median_of_scene_rotation_medians_degrees": float(np.median(rotations)),
            "median_of_scene_translation_medians_degrees": float(
                np.median(translations)
            ),
            "median_of_scene_depth_mean_abs_rel": float(np.median(depths)),
            "scenes_over_rotation_threshold": int(
                np.count_nonzero(rotations > rotation_threshold)
            ),
            "scenes_over_depth_threshold": int(
                np.count_nonzero(depths > depth_threshold)
            ),
        }
    return {
        "rotation_threshold_degrees": rotation_threshold,
        "depth_abs_rel_threshold": depth_threshold,
        "comparison_count": len(rows),
        "comparisons": rows,
        "by_variant": by_variant,
    }


def summarize_eth3d_evaluations(paths: list[Path]) -> dict[str, object]:
    """Aggregate GT metrics and compute absolute deltas from identity."""

    if not paths:
        raise ValueError("at least one ETH3D evaluation record is required")
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        depth = record["depth"]
        rows.append(
            {
                "variant": record.get("variant", Path(record["prediction"]).stem),
                "rotation_median_degrees": record["relative_rotation_degrees"][
                    "median"
                ],
                "translation_median_degrees": record[
                    "translation_direction_degrees"
                ]["median"],
                "depth_mean_abs_rel": depth["mean_abs_rel"] if depth else None,
                "valid_depth_pixels": depth["valid_pixels"] if depth else None,
                "source": str(path),
            }
        )
    identities = [row for row in rows if row["variant"] == "identity"]
    if len(identities) != 1:
        raise ValueError("ETH3D summary requires exactly one identity record")
    identity = identities[0]
    for row in rows:
        row["rotation_delta_from_identity_degrees"] = float(
            row["rotation_median_degrees"] - identity["rotation_median_degrees"]
        )
        row["translation_delta_from_identity_degrees"] = float(
            row["translation_median_degrees"]
            - identity["translation_median_degrees"]
        )
        if row["depth_mean_abs_rel"] is None:
            row["depth_abs_rel_delta_from_identity"] = None
        else:
            row["depth_abs_rel_delta_from_identity"] = float(
                row["depth_mean_abs_rel"] - identity["depth_mean_abs_rel"]
            )
    return {
        "evaluation_count": len(rows),
        "depth_evaluated": identity["depth_mean_abs_rel"] is not None,
        "identity": identity,
        "evaluations": rows,
    }
