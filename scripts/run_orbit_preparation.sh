#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != "eth3d" && $1 != "dtu" ) ]]; then
  echo "usage: $0 {eth3d|dtu}" >&2
  exit 2
fi

dataset="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
protocol="${repo_root}/configs/orbit_projection_protocol.json"
cd "${repo_root}"

case "${dataset}" in
  eth3d)
    selection_report="/mnt/e/camcanon3r-data/eth3d_selected/selection_report.json"
    canonical_root="data/eth3d_training/raw_canonical"
    orbit_root="data/eth3d_training/raw_canonical_orbit"
    preparation_report="results/orbit/eth3d_preparation.json"
    audit_report="results/orbit/eth3d_preparation_audit.json"
    expected_scenes=13
    expected_images=468
    mapfile -t scenes < <(
      "${python_bin}" -c \
        'import json,sys; print(*(x["scene"] for x in json.load(open(sys.argv[1]))["selection"]["scenes"]), sep="\n")' \
        "${selection_report}"
    )
    ;;
  dtu)
    dtu_protocol="configs/dtu_protocol.json"
    canonical_root="data/dtu/rectified_canonical"
    orbit_root="data/dtu/rectified_canonical_orbit"
    preparation_report="results/orbit/dtu_preparation.json"
    audit_report="results/orbit/dtu_preparation_audit.json"
    expected_scenes=22
    expected_images=594
    mapfile -t scenes < <(
      "${python_bin}" -c \
        'import json,sys; print(*(f"scan{x}" for x in json.load(open(sys.argv[1]))["evaluation_scans"]), sep="\n")' \
        "${dtu_protocol}"
    )
    ;;
esac

if [[ ${#scenes[@]} -ne ${expected_scenes} ]]; then
  echo "canonical orbit scene design is incomplete" >&2
  exit 1
fi

PYTHONPATH=src "${python_bin}" scripts/prepare_orbit_sweep.py \
  "${canonical_root}" "${orbit_root}" --protocol "${protocol}" \
  --scenes "${scenes[@]}" --report "${preparation_report}" --resume

PYTHONPATH=src "${python_bin}" scripts/audit_orbit_sweep.py \
  "${canonical_root}" "${orbit_root}" --protocol "${protocol}" \
  --scenes "${scenes[@]}" --output "${audit_report}"

"${python_bin}" - "${audit_report}" "${expected_scenes}" "${expected_images}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "complete"
assert report["scene_count"] == int(sys.argv[2])
assert report["image_count"] == int(sys.argv[3])
assert report["mask_count"] == int(sys.argv[3])
assert report["decoded_rgb_matches"] == int(sys.argv[3])
assert report["decoded_mask_matches"] == int(sys.argv[3])
print(json.dumps({
    "status": report["status"],
    "scene_count": report["scene_count"],
    "image_count": report["image_count"],
    "tree_sha256": report["tree_sha256"],
}))
PY
