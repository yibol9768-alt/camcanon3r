"""Deterministic selection from official ETH3D training archives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def parse_7z_slt(text: str) -> dict[str, int]:
    """Parse file sizes from ``7z l -slt`` output."""

    members: dict[str, int] = {}
    record: dict[str, str] = {}
    in_members = False

    def commit() -> None:
        if not record or record.get("Folder") == "+":
            return
        path = record.get("Path")
        size = record.get("Size")
        if path is None or size is None:
            return
        normalized = path.replace("\\", "/")
        if normalized in members:
            raise ValueError(f"duplicate archive member: {normalized}")
        members[normalized] = int(size)

    for raw_line in text.splitlines():
        line = raw_line.strip("\r")
        if line == "----------":
            in_members = True
            record = {}
            continue
        if not in_members:
            continue
        if not line:
            commit()
            record = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            record[key] = value
    commit()
    if not members:
        raise ValueError("7z listing contained no file members")
    return members


def build_eth3d_selection(
    *,
    scenes: Iterable[str],
    undistorted_members: Mapping[str, int],
    raw_members: Mapping[str, int],
    depth_members: Mapping[str, Mapping[str, int]],
    views_per_scene: int,
) -> dict[str, object]:
    """Select the first sorted DSLR names per frozen scene."""

    if views_per_scene < 2:
        raise ValueError("at least two views per scene are required")
    scene_names = list(scenes)
    if len(scene_names) != len(set(scene_names)):
        raise ValueError("scene names must be unique")
    if set(depth_members) != set(scene_names):
        missing = sorted(set(scene_names) - set(depth_members))
        extra = sorted(set(depth_members) - set(scene_names))
        raise ValueError(f"depth archive scene mismatch: missing={missing}, extra={extra}")

    selected_scenes: list[dict[str, object]] = []
    for scene in scene_names:
        undistorted_prefix = f"{scene}/images/dslr_images_undistorted/"
        names = sorted(
            path.removeprefix(undistorted_prefix)
            for path in undistorted_members
            if path.startswith(undistorted_prefix)
            and "/" not in path.removeprefix(undistorted_prefix)
            and path.lower().endswith((".jpg", ".jpeg"))
        )
        if len(names) < views_per_scene:
            raise ValueError(
                f"scene {scene!r} has only {len(names)} undistorted DSLR images"
            )
        selected_names = names[:views_per_scene]
        raw_paths = [f"{scene}/images/dslr_images/{name}" for name in selected_names]
        undistorted_paths = [
            f"{scene}/images/dslr_images_undistorted/{name}"
            for name in selected_names
        ]
        selected_depth_members = depth_members[scene]
        depth_paths = [
            f"{scene}/ground_truth_depth/dslr_images/{name}"
            for name in selected_names
        ]
        calibration_raw = [
            f"{scene}/dslr_calibration_jpg/{filename}"
            for filename in ("cameras.txt", "images.txt", "points3D.txt")
        ]
        calibration_undistorted = [
            f"{scene}/dslr_calibration_undistorted/{filename}"
            for filename in ("cameras.txt", "images.txt", "points3D.txt")
        ]
        _require_members(raw_members, raw_paths + calibration_raw, scene, "raw")
        _require_members(
            undistorted_members,
            undistorted_paths + calibration_undistorted,
            scene,
            "undistorted",
        )
        _require_members(
            selected_depth_members, depth_paths, scene, "rendered depth"
        )
        selected_scenes.append(
            {
                "scene": scene,
                "image_names": selected_names,
                "raw": _records(raw_paths, raw_members),
                "undistorted": _records(
                    undistorted_paths, undistorted_members
                ),
                "depth": _records(depth_paths, selected_depth_members),
                "raw_calibration": _records(calibration_raw, raw_members),
                "undistorted_calibration": _records(
                    calibration_undistorted, undistorted_members
                ),
            }
        )
    return {
        "selection_policy": (
            "first lexicographically sorted DSLR filenames per scene, frozen "
            "without model outcomes"
        ),
        "views_per_scene": views_per_scene,
        "scenes": selected_scenes,
    }


def selected_paths(
    selection: Mapping[str, object], field: str
) -> list[str]:
    paths: list[str] = []
    for scene in selection["scenes"]:
        if not isinstance(scene, dict):
            raise TypeError("selection scenes must be dictionaries")
        records = scene[field]
        if not isinstance(records, list):
            raise TypeError(f"selection field {field!r} must be a list")
        paths.extend(str(record["path"]) for record in records)
    return paths


def _records(
    paths: Iterable[str], members: Mapping[str, int]
) -> list[dict[str, object]]:
    return [{"path": path, "bytes": int(members[path])} for path in paths]


def _require_members(
    members: Mapping[str, int],
    paths: Iterable[str],
    scene: str,
    label: str,
) -> None:
    missing = [path for path in paths if path not in members]
    if missing:
        raise ValueError(
            f"scene {scene!r} is missing {label} archive members: {missing}"
        )
