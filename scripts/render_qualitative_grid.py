#!/usr/bin/env python3
"""Render the frozen multi-scene, multi-model qualitative point-map grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from camcanon3r.dtu import read_dtu_projection
from camcanon3r.eth3d import (
    colmap_camera_matrix,
    read_colmap_cameras,
    read_colmap_images,
)
from camcanon3r.qualitative import (
    apply_camera_pose_alignment,
    median_camera_baseline,
    rasterize_aligned_points,
    source_supported_prediction_points,
)

VARIANTS = (
    ("identity", "main", "Identity"),
    ("asymmetric_crop_075", "main", "75% crop"),
    ("canonical_asymmetric_crop_075", "repair", "Canonical"),
)
MODEL_DISPLAY = {"vggt": "VGGT", "dust3r": "DUSt3R"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("eth3d-training-raw", "dtu-held-out"))
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/qualitative_protocol.json")
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=5,
        metavar=(
            "NAME",
            "MAIN_PREDICTIONS",
            "MAIN_RESULTS",
            "REPAIR_PREDICTIONS",
            "REPAIR_RESULTS",
        ),
        required=True,
    )
    parser.add_argument("--repair-prepared-root", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-points-per-view", type=int, default=25_000)
    parser.add_argument("--raster-width", type=int, default=320)
    parser.add_argument("--raster-height", type=int, default=240)
    parser.add_argument("--log-depth-min", type=float, default=-1.0)
    parser.add_argument("--log-depth-max", type=float, default=2.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"qualitative JSON source is not an object: {path}")
    return payload


def _target_cameras(
    dataset: str, result: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    calibration_dir = Path(str(result["calibration_dir"]))
    inputs = [str(value) for value in result["inputs"]]
    if dataset == "dtu-held-out":
        camera_ids = [int(value) for value in result["camera_ids"]]
        cameras = [
            read_dtu_projection(calibration_dir / f"pos_{camera_id:03d}.txt")
            for camera_id in camera_ids
        ]
        sizes = [tuple(int(value) for value in size) for size in result["camera_sizes"]]
    else:
        camera_by_id = read_colmap_cameras(calibration_dir / "cameras.txt")
        image_by_stem = read_colmap_images(calibration_dir / "images.txt")
        image_records = [image_by_stem[Path(name).stem] for name in inputs]
        target_cameras = [camera_by_id[record.camera_id] for record in image_records]
        cameras = [
            (colmap_camera_matrix(camera), image_record.extrinsic)
            for camera, image_record in zip(target_cameras, image_records, strict=True)
        ]
        sizes = [(camera.width, camera.height) for camera in target_cameras]
    intrinsics = np.stack([camera[0] for camera in cameras])
    extrinsics = np.stack([camera[1] for camera in cameras])
    if len(intrinsics) != len(inputs) or len(sizes) != len(inputs):
        raise ValueError("qualitative target-camera design does not match inputs")
    return intrinsics, extrinsics, sizes


def _point_metric_text(dataset: str, result: dict[str, Any]) -> str:
    rotation = float(result["relative_rotation_degrees"]["median"])
    point_cloud = result.get("point_cloud")
    if not isinstance(point_cloud, dict) or point_cloud.get("status") != "available":
        status = (
            point_cloud.get("status", "unavailable") if point_cloud else "unavailable"
        )
        return f"R {rotation:.2f}° | A/C {status}"
    if dataset == "dtu-held-out":
        accuracy = float(point_cloud["accuracy_millimeters"]["mean"])
        completeness = float(point_cloud["completeness_millimeters"]["mean"])
        unit = "mm"
    else:
        accuracy = 100.0 * float(point_cloud["accuracy_meters"]["mean"])
        completeness = 100.0 * float(point_cloud["completeness_meters"]["mean"])
        unit = "cm"
    return f"R {rotation:.2f}° | A/C {accuracy:.1f}/{completeness:.1f} {unit}"


def _panel(
    dataset: str,
    scene: str,
    variant: str,
    prediction_path: Path,
    result_path: Path,
    *,
    repair_prepared_root: Path,
    maximum_points_per_view: int,
    output_size: tuple[int, int],
) -> dict[str, Any]:
    metadata_path = prediction_path.with_suffix(".json")
    for path in (prediction_path, metadata_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"qualitative source is missing: {path}")
    metadata = _read_json(metadata_path)
    result = _read_json(result_path)
    if result.get("scene") != scene or result.get("variant") != variant:
        raise ValueError(f"qualitative result identity mismatch: {result_path}")
    if result.get("prediction") != str(prediction_path.resolve()):
        raise ValueError(
            f"qualitative result does not reference expected prediction: {result_path}"
        )
    inputs = [str(value) for value in metadata.get("inputs", [])]
    transforms = metadata.get("spatial_transforms")
    if (
        len(inputs) < 2
        or not isinstance(transforms, list)
        or [record.get("input") for record in transforms] != inputs
    ):
        raise ValueError(f"qualitative prediction metadata is invalid: {metadata_path}")
    if result.get("inputs") != inputs:
        raise ValueError(
            f"qualitative evaluation inputs do not match prediction: {result_path}"
        )
    source_sizes = [
        tuple(int(value) for value in record["input_size"]) for record in transforms
    ]

    point_cloud = result.get("point_cloud")
    status = (
        str(point_cloud.get("status"))
        if isinstance(point_cloud, dict)
        else "unavailable"
    )
    raster: np.ndarray | None = None
    render_status = status
    if status == "available":
        with np.load(prediction_path, allow_pickle=False) as prediction:
            points = source_supported_prediction_points(
                prediction["world_points"],
                prediction["source_to_model_affine"],
                source_sizes,
                maximum_per_view=maximum_points_per_view,
            )
        alignment = point_cloud.get("alignment")
        if not isinstance(alignment, dict):
            raise ValueError(
                f"qualitative result has no point alignment: {result_path}"
            )
        aligned = apply_camera_pose_alignment(points, alignment)
        intrinsics, extrinsics, target_sizes = _target_cameras(dataset, result)
        baseline = median_camera_baseline(extrinsics)
        try:
            raster = rasterize_aligned_points(
                aligned,
                intrinsics[0],
                extrinsics[0],
                target_sizes[0],
                output_size=output_size,
                baseline=baseline,
            )
        except ValueError as error:
            render_status = f"unrenderable: {error}"

    mask_path: Path | None = None
    mask: np.ndarray | None = None
    if variant == "canonical_asymmetric_crop_075":
        mask_path = (
            repair_prepared_root / scene / "_masks" / variant / Path(inputs[0]).name
        )
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"qualitative validity mask is missing: {mask_path}"
            )
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened.convert("L"), dtype=np.uint8)
        if not np.isin(mask, (0, 255)).all():
            raise ValueError(f"qualitative validity mask is not binary: {mask_path}")

    sources = {
        "prediction": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "evaluation": str(result_path.resolve()),
        "evaluation_sha256": _sha256(result_path),
        "validity_mask": str(mask_path.resolve()) if mask_path else None,
        "validity_mask_sha256": _sha256(mask_path) if mask_path else None,
    }
    return {
        "scene": scene,
        "variant": variant,
        "raster": raster,
        "mask": mask,
        "render_status": render_status,
        "metric_text": _point_metric_text(dataset, result),
        "sources": sources,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.log_depth_min >= args.log_depth_max:
        raise ValueError("qualitative log-depth range must be increasing")
    if args.raster_width <= 0 or args.raster_height <= 0:
        raise ValueError("qualitative raster dimensions must be positive")
    protocol = _read_json(args.protocol)
    expected_models = [str(value) for value in protocol["models"]]
    model_records = {
        str(record[0]): {
            "main_predictions": Path(record[1]),
            "main_results": Path(record[2]),
            "repair_predictions": Path(record[3]),
            "repair_results": Path(record[4]),
        }
        for record in args.model
    }
    if len(model_records) != len(args.model):
        raise ValueError("qualitative model arguments contain a duplicate name")
    if list(model_records) != expected_models:
        raise ValueError(
            "qualitative model order does not match frozen protocol: "
            f"expected={expected_models}, actual={list(model_records)}"
        )
    dataset_record = protocol["datasets"][args.dataset]
    scenes = [str(value) for value in dataset_record["selected_scenes"]]
    if len(scenes) != 4 or len(set(scenes)) != len(scenes):
        raise ValueError("qualitative protocol requires four unique selected scenes")
    if [variant for variant, _, _ in VARIANTS] != protocol["variants"]:
        raise ValueError("qualitative variant order does not match frozen protocol")
    display_contract = protocol["display_contract"]
    actual_rendering = {
        "maximum_points_per_view": args.maximum_points_per_view,
        "raster_size": [args.raster_width, args.raster_height],
        "colormap": "viridis",
        "log10_depth_per_baseline_range": [
            args.log_depth_min,
            args.log_depth_max,
        ],
    }
    expected_rendering = {key: display_contract[key] for key in actual_rendering}
    if actual_rendering != expected_rendering:
        raise ValueError(
            "qualitative rendering parameters do not match frozen protocol: "
            f"expected={expected_rendering}, actual={actual_rendering}"
        )

    panels: dict[tuple[str, str, str], dict[str, Any]] = {}
    for scene in scenes:
        for model in expected_models:
            roots = model_records[model]
            for variant, source, _ in VARIANTS:
                prediction_root = roots[f"{source}_predictions"]
                result_root = roots[f"{source}_results"]
                panels[(scene, model, variant)] = _panel(
                    args.dataset,
                    scene,
                    variant,
                    prediction_root / scene / f"{variant}.npz",
                    result_root / scene / f"{variant}_vs_gt.json",
                    repair_prepared_root=args.repair_prepared_root,
                    maximum_points_per_view=args.maximum_points_per_view,
                    output_size=(args.raster_width, args.raster_height),
                )

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(scenes),
        len(expected_models) * len(VARIANTS),
        figsize=(7.05, 4.85),
        squeeze=False,
    )
    colormap = mpl.colormaps["viridis"].copy()
    colormap.set_bad("#f3f3f3")
    for row, scene in enumerate(scenes):
        for model_index, model in enumerate(expected_models):
            for variant_index, (variant, _, label) in enumerate(VARIANTS):
                column = model_index * len(VARIANTS) + variant_index
                axis = axes[row][column]
                panel = panels[(scene, model, variant)]
                raster = panel["raster"]
                if raster is None:
                    axis.set_facecolor("#f3f3f3")
                    axis.text(
                        0.5,
                        0.55,
                        str(panel["render_status"]),
                        ha="center",
                        va="center",
                        fontsize=4.5,
                        wrap=True,
                        transform=axis.transAxes,
                    )
                else:
                    axis.imshow(
                        raster,
                        cmap=colormap,
                        vmin=args.log_depth_min,
                        vmax=args.log_depth_max,
                        interpolation="nearest",
                    )
                axis.text(
                    0.5,
                    0.015,
                    str(panel["metric_text"]),
                    ha="center",
                    va="bottom",
                    fontsize=4.25,
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.7,
                    },
                    transform=axis.transAxes,
                )
                if panel["mask"] is not None:
                    inset = axis.inset_axes([0.76, 0.70, 0.22, 0.27])
                    inset.imshow(panel["mask"], cmap="gray", vmin=0, vmax=255)
                    inset.set_xticks([])
                    inset.set_yticks([])
                    for spine in inset.spines.values():
                        spine.set_color("#F58518")
                        spine.set_linewidth(0.8)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.35)
                if row == 0:
                    model_label = MODEL_DISPLAY.get(model, model)
                    axis.set_title(f"{model_label}\n{label}", fontsize=6.2, pad=2)
                if column == 0:
                    axis.set_ylabel(scene, fontsize=6.2, labelpad=2)

    normalization = mpl.colors.Normalize(
        vmin=args.log_depth_min, vmax=args.log_depth_max
    )
    colorbar_axis = figure.add_axes([0.20, 0.045, 0.60, 0.018])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=normalization, cmap=colormap),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(
        r"Aligned target-camera depth, $\log_{10}(z / \mathrm{median\ baseline})$",
        fontsize=6,
    )
    colorbar.ax.tick_params(labelsize=5.5)
    figure.suptitle(
        (
            "ETH3D raw — frozen qualitative scenes"
            if args.dataset == "eth3d-training-raw"
            else "DTU held-out — frozen qualitative scenes"
        ),
        fontsize=8,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.07, right=0.995, top=0.91, bottom=0.12, wspace=0.035, hspace=0.08
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.png, dpi=240, bbox_inches="tight")
    plt.close(figure)

    source_records = [
        {
            "model": model,
            **panels[(scene, model, variant)]["sources"],
            "scene": scene,
            "variant": variant,
            "render_status": panels[(scene, model, variant)]["render_status"],
        }
        for scene in scenes
        for model in expected_models
        for variant, _, _ in VARIANTS
    ]
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "dataset": args.dataset,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "models": expected_models,
        "scenes": scenes,
        "variants": [variant for variant, _, _ in VARIANTS],
        "rendering_contract": {
            "alignment": "evaluator camera-pose-only orientation-preserving Sim(3)",
            "projection": "first frozen target camera with deterministic z-buffer",
            "depth_normalization": "median pairwise frozen target-camera baseline",
            "maximum_points_per_view": args.maximum_points_per_view,
            "raster_size": [args.raster_width, args.raster_height],
            "colormap": "viridis",
            "log10_depth_per_baseline_range": [
                args.log_depth_min,
                args.log_depth_max,
            ],
            "per_scene_axis_or_colormap_tuning": False,
            "canonical_validity_mask_inset": True,
        },
        "sources": source_records,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "png": str(args.png.resolve()) if args.png else None,
        "png_sha256": _sha256(args.png) if args.png else None,
    }
    _write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "panel_count": len(source_records),
                "output": str(args.output),
                "report": str(args.report),
            }
        )
    )


if __name__ == "__main__":
    main()
