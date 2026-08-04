"""Planning helpers for resumable multi-scene inference sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepRun:
    scene: str
    variant: str
    prepared_dir: Path
    output: Path
    skip: bool


def plan_prediction_sweep(
    prepared_root: Path,
    output_root: Path,
    *,
    variants: list[str],
    scenes: list[str] | None,
    resume: bool,
    overwrite: bool,
) -> list[SweepRun]:
    """Validate all inputs and decide which prediction pairs should run."""

    if not variants:
        raise ValueError("at least one variant is required")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")

    scene_names: list[str | None] = scenes if scenes else [None]
    planned: list[SweepRun] = []
    for scene_name in scene_names:
        scene_root = prepared_root / scene_name if scene_name else prepared_root
        scene_output = output_root / scene_name if scene_name else output_root
        label = scene_name or prepared_root.name
        for variant in variants:
            prepared_dir = scene_root / variant
            if not prepared_dir.is_dir():
                raise FileNotFoundError(f"prepared variant is missing: {prepared_dir}")
            output = scene_output / f"{variant}.npz"
            npz_exists = output.exists()
            json_exists = output.with_suffix(".json").exists()
            if npz_exists != json_exists and not (resume or overwrite):
                raise RuntimeError(
                    "partial prediction output requires --resume or --overwrite: "
                    f"{output}"
                )
            complete = npz_exists and json_exists
            if complete and not (resume or overwrite):
                raise FileExistsError(
                    f"prediction already exists; use --resume or --overwrite: {output}"
                )
            planned.append(
                SweepRun(
                    scene=label,
                    variant=variant,
                    prepared_dir=prepared_dir,
                    output=output,
                    skip=complete and resume,
                )
            )
    return planned


def plan_vggt_sweep(
    prepared_root: Path,
    output_root: Path,
    *,
    variants: list[str],
    scenes: list[str] | None,
    resume: bool,
    overwrite: bool,
) -> list[SweepRun]:
    """Backward-compatible alias for the model-neutral sweep planner."""

    return plan_prediction_sweep(
        prepared_root,
        output_root,
        variants=variants,
        scenes=scenes,
        resume=resume,
        overwrite=overwrite,
    )
