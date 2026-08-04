from __future__ import annotations

import json
from pathlib import Path

import pytest

from camcanon3r.dtu_selection import build_dtu_remote_selections

SCANS = (
    1,
    4,
    9,
    10,
    11,
    12,
    13,
    15,
    23,
    24,
    29,
    32,
    33,
    34,
    48,
    49,
    62,
    75,
    77,
    110,
    114,
    118,
)
CAMERAS = (23, 26, 29)


def _record(path: str) -> dict[str, object]:
    return {"path": path, "bytes": 7, "compressed_bytes": 5, "crc32": "1234abcd"}


def _index(path: Path, source: dict[str, object], members: list[str]) -> None:
    records = [_record(member) for member in members]
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "url": source["url"],
                "expected_bytes": source["expected_bytes"],
                "etag": f'"{source["etag"]}"',
                "match_count": len(records),
                "matches": records,
                "matches_truncated": False,
            }
        ),
        encoding="utf-8",
    )


def _design(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sources = {
        "frozen_before_dtu_gt_inspection": True,
        "archives": {
            name: {
                "url": f"https://example.test/{name}.zip",
                "expected_bytes": 100 + index,
                "etag": f"etag-{name}",
                "index_filename": f"index_{name}.json",
            }
            for index, name in enumerate(("rectified", "sampleset", "points"))
        },
    }
    protocol = {
        "frozen_before_dtu_gt_inspection": True,
        "evaluation_scans": list(SCANS),
        "rectified_archive_camera_ids_one_based": list(CAMERAS),
        "lighting_index": 3,
    }
    source_path = tmp_path / "sources.json"
    protocol_path = tmp_path / "protocol.json"
    source_path.write_text(json.dumps(sources), encoding="utf-8")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    rectified = [
        f"Rectified/scan{scan}/rect_{camera:03d}_3_r5000.png"
        for scan in SCANS
        for camera in CAMERAS
    ]
    static = [
        "SampleSet/ReadMe.txt",
        "SampleSet/MVS Data/Calibration/cal18/calib_data.mat",
        "SampleSet/MVS Data/Calibration/cal18/Calib_Results_left.mat",
        "SampleSet/MVS Data/Calibration/cal18/Calib_Results_stereo.mat",
        "SampleSet/Matlab evaluation code/BaseEvalMain_web.m",
        "SampleSet/Matlab evaluation code/BaseEval2Obj_web.m",
        "SampleSet/Matlab evaluation code/ComputeStat_web.m",
        "SampleSet/Matlab evaluation code/MaxDistCP.m",
        "SampleSet/Matlab evaluation code/PointCompareMain.m",
        "SampleSet/Matlab evaluation code/reducePts_haa.m",
        "SampleSet/Matlab evaluation code/plyread.m",
    ]
    sampleset = static + [
        f"SampleSet/MVS Data/Calibration/cal18/pos_{camera:03d}.txt"
        for camera in CAMERAS
    ]
    sampleset += [
        f"SampleSet/MVS Data/ObsMask/{name}"
        for scan in SCANS
        for name in (f"ObsMask{scan}_10.mat", f"Plane{scan}.mat")
    ]
    points = [f"Points/stl/stl{scan:03d}_total.ply" for scan in SCANS]
    index_root = tmp_path / "indexes"
    index_root.mkdir()
    for name, members in (
        ("rectified", rectified),
        ("sampleset", sampleset),
        ("points", points),
    ):
        _index(index_root / f"index_{name}.json", sources["archives"][name], members)
    return source_path, protocol_path, index_root, tmp_path / "output"


def test_build_dtu_remote_selections_freezes_complete_design(tmp_path: Path) -> None:
    source, protocol, indexes, output = _design(tmp_path)
    summary = build_dtu_remote_selections(source, protocol, indexes, output)
    assert summary["member_counts"] == {
        "rectified": 66,
        "sampleset": 58,
        "points": 22,
    }
    rectified = json.loads((output / "rectified.json").read_text())
    assert rectified["members"][0]["target"] == ("source/scan1/rect_023_3_r5000.png")
    assert rectified["members"][-1]["source"] == (
        "Rectified/scan118/rect_029_3_r5000.png"
    )


def test_build_dtu_remote_selections_rejects_missing_member(tmp_path: Path) -> None:
    source, protocol, indexes, output = _design(tmp_path)
    path = indexes / "index_points.json"
    payload = json.loads(path.read_text())
    payload["matches"].pop()
    payload["match_count"] -= 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="stl118_total"):
        build_dtu_remote_selections(source, protocol, indexes, output)
