"""Freeze the exact remote DTU subset from audited ZIP indexes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .prediction import write_json_atomic


def _quoted_etag(value: object) -> str:
    etag = str(value)
    return etag if etag.startswith(('"', 'W/"')) else f'"{etag}"'


def _load_index(
    index_path: Path, source: Mapping[str, object]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "url": str(source["url"]),
        "expected_bytes": int(source["expected_bytes"]),
        "etag": _quoted_etag(source["etag"]),
        "matches_truncated": False,
    }
    for field, value in expected.items():
        if index.get(field) != value:
            raise ValueError(
                f"DTU ZIP index identity mismatch for {index_path}: "
                f"field={field}, expected={value!r}, actual={index.get(field)!r}"
            )
    matches = index.get("matches")
    if not isinstance(matches, list) or len(matches) != index.get("match_count"):
        raise ValueError(f"DTU ZIP index is incomplete: {index_path}")
    records: dict[str, dict[str, Any]] = {}
    for record in matches:
        path = str(record["path"])
        if path in records:
            raise ValueError(f"duplicate DTU ZIP member in index: {path}")
        records[path] = record
    return index, records


def _member(
    records: Mapping[str, Mapping[str, object]], source: str, target: str
) -> dict[str, object]:
    if source not in records:
        raise FileNotFoundError(f"required DTU ZIP member is missing: {source}")
    record = records[source]
    return {
        "source": source,
        "target": target,
        "bytes": int(record["bytes"]),
        "crc32": str(record["crc32"]),
    }


def _manifest(
    source: Mapping[str, object],
    members: Sequence[dict[str, object]],
    *,
    archive_id: str,
    protocol_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "archive_id": archive_id,
        "protocol": str(protocol_path),
        "url": str(source["url"]),
        "expected_bytes": int(source["expected_bytes"]),
        "etag": str(source["etag"]),
        "members": list(members),
    }


def build_dtu_remote_selections(
    source_path: Path,
    protocol_path: Path,
    index_root: Path,
    output_root: Path,
) -> dict[str, object]:
    sources_payload = json.loads(source_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not sources_payload.get("frozen_before_dtu_gt_inspection"):
        raise ValueError("DTU source identities were not frozen before GT inspection")
    if not protocol.get("frozen_before_dtu_gt_inspection"):
        raise ValueError("DTU protocol was not frozen before GT inspection")
    sources = sources_payload.get("archives")
    if not isinstance(sources, Mapping):
        raise TypeError("DTU source config must contain an archives object")
    scans = tuple(int(value) for value in protocol["evaluation_scans"])
    camera_ids = tuple(
        int(value) for value in protocol["rectified_archive_camera_ids_one_based"]
    )
    lighting = int(protocol["lighting_index"])
    if len(scans) != 22 or len(set(scans)) != 22:
        raise ValueError("DTU protocol requires 22 unique evaluation scans")
    if len(camera_ids) != 3 or len(set(camera_ids)) != 3:
        raise ValueError("DTU protocol requires three unique camera IDs")

    indexes: dict[str, dict[str, dict[str, Any]]] = {}
    for archive_id in ("rectified", "sampleset", "points"):
        source = sources[archive_id]
        if not isinstance(source, Mapping):
            raise TypeError(f"DTU source record must be an object: {archive_id}")
        _, indexes[archive_id] = _load_index(
            index_root / str(source["index_filename"]), source
        )

    rectified_members = [
        _member(
            indexes["rectified"],
            (f"Rectified/scan{scan}/rect_{camera_id:03d}_{lighting}_r5000.png"),
            (f"source/scan{scan}/rect_{camera_id:03d}_{lighting}_r5000.png"),
        )
        for scan in scans
        for camera_id in camera_ids
    ]
    sampleset_static = (
        ("SampleSet/ReadMe.txt", "official/ReadMe.txt"),
        (
            "SampleSet/MVS Data/Calibration/cal18/calib_data.mat",
            "calibration/cal18/calib_data.mat",
        ),
        (
            "SampleSet/MVS Data/Calibration/cal18/Calib_Results_left.mat",
            "calibration/cal18/Calib_Results_left.mat",
        ),
        (
            "SampleSet/MVS Data/Calibration/cal18/Calib_Results_stereo.mat",
            "calibration/cal18/Calib_Results_stereo.mat",
        ),
        (
            "SampleSet/Matlab evaluation code/BaseEvalMain_web.m",
            "official/matlab_evaluation/BaseEvalMain_web.m",
        ),
        (
            "SampleSet/Matlab evaluation code/BaseEval2Obj_web.m",
            "official/matlab_evaluation/BaseEval2Obj_web.m",
        ),
        (
            "SampleSet/Matlab evaluation code/ComputeStat_web.m",
            "official/matlab_evaluation/ComputeStat_web.m",
        ),
        (
            "SampleSet/Matlab evaluation code/MaxDistCP.m",
            "official/matlab_evaluation/MaxDistCP.m",
        ),
        (
            "SampleSet/Matlab evaluation code/PointCompareMain.m",
            "official/matlab_evaluation/PointCompareMain.m",
        ),
        (
            "SampleSet/Matlab evaluation code/reducePts_haa.m",
            "official/matlab_evaluation/reducePts_haa.m",
        ),
        (
            "SampleSet/Matlab evaluation code/plyread.m",
            "official/matlab_evaluation/plyread.m",
        ),
    )
    sampleset_members = [
        _member(indexes["sampleset"], source, target)
        for source, target in sampleset_static
    ]
    sampleset_members.extend(
        _member(
            indexes["sampleset"],
            f"SampleSet/MVS Data/Calibration/cal18/pos_{camera_id:03d}.txt",
            f"calibration/cal18/pos_{camera_id:03d}.txt",
        )
        for camera_id in camera_ids
    )
    for scan in scans:
        for filename in (f"ObsMask{scan}_10.mat", f"Plane{scan}.mat"):
            sampleset_members.append(
                _member(
                    indexes["sampleset"],
                    f"SampleSet/MVS Data/ObsMask/{filename}",
                    f"gt/ObsMask/{filename}",
                )
            )
    points_members = [
        _member(
            indexes["points"],
            f"Points/stl/stl{scan:03d}_total.ply",
            f"gt/Points/stl/stl{scan:03d}_total.ply",
        )
        for scan in scans
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    member_counts: dict[str, int] = {}
    for archive_id, members in (
        ("rectified", rectified_members),
        ("sampleset", sampleset_members),
        ("points", points_members),
    ):
        source = sources[archive_id]
        assert isinstance(source, Mapping)
        manifest = _manifest(
            source,
            members,
            archive_id=archive_id,
            protocol_path=protocol_path,
        )
        write_json_atomic(output_root / f"{archive_id}.json", manifest)
        member_counts[archive_id] = len(members)
    summary = {
        "status": "complete",
        "scan_count": len(scans),
        "camera_count": len(camera_ids),
        "member_counts": member_counts,
        "output_root": str(output_root),
    }
    return summary
