#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != "eth3d" && $1 != "dtu" ) ]]; then
  echo "usage: $0 {eth3d|dtu}" >&2
  exit 2
fi

dataset="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
variant_config="${repo_root}/configs/support_control_variants.json"

cd "${repo_root}"
mapfile -t variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["ordered_variants"], sep="\n")' \
    "${variant_config}"
)
if [[ ${#variants[@]} -ne 3 ]]; then
  echo "support control must contain exactly three variants" >&2
  exit 1
fi

case "${dataset}" in
  eth3d)
    selection_root="/mnt/e/camcanon3r-data/eth3d_selected"
    prepared_root="data/eth3d_training/raw_support_control"
    reference_root="data/eth3d_training/raw_mechanism"
    audit_output="results/eth3d_training/support_control_preparation_audit.json"
    PYTHONPATH=src "${python_bin}" scripts/prepare_eth3d_selection.py \
      "${selection_root}" "${prepared_root}" --domain raw \
      --variants "${variants[@]}" --seed 17 --resume
    PYTHONPATH=src "${python_bin}" scripts/audit_support_control.py \
      "${prepared_root}" "${reference_root}" \
      --variant-config "${variant_config}" \
      --eth3d-selection-report "${selection_root}/selection_report.json" \
      --output "${audit_output}" >/dev/null
    expected_scenes=13
    expected_images=156
    expected_anchor_matches=52
    ;;
  dtu)
    archive_root="/mnt/e/camcanon3r-data/dtu_mvs"
    selection_root="/mnt/e/camcanon3r-data/dtu_selected"
    prepared_root="data/dtu/rectified_support_control"
    reference_root="data/dtu/rectified_mechanism"
    protocol="configs/dtu_support_control_protocol.json"
    audit_output="results/dtu/support_control_content_audit.json"
    for name in sampleset rectified points; do
      "${python_bin}" - "${archive_root}/${name}_extraction_report.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "complete" or report.get("completed_members") != report.get("member_count"):
    raise RuntimeError(f"DTU extraction is incomplete: {path}")
PY
    done
    PYTHONPATH=src "${python_bin}" scripts/prepare_dtu_selection.py \
      "${selection_root}" "${prepared_root}" \
      "${archive_root}/rectified_extraction_report.json" \
      --protocol "${protocol}" --variant-config "${variant_config}" --resume
    PYTHONPATH=src "${python_bin}" scripts/audit_dtu_mechanism.py \
      "${prepared_root}" --protocol "${protocol}" \
      --variant-config "${variant_config}" \
      --output results/dtu/support_control_preparation_audit.json >/dev/null
    PYTHONPATH=src "${python_bin}" scripts/audit_support_control.py \
      "${prepared_root}" "${reference_root}" \
      --variant-config "${variant_config}" --dtu-protocol "${protocol}" \
      --output "${audit_output}" >/dev/null
    expected_scenes=22
    expected_images=198
    expected_anchor_matches=66
    ;;
esac

"${python_bin}" - "${audit_output}" "${dataset}" "${expected_scenes}" \
  "${expected_images}" "${expected_anchor_matches}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_scenes, expected_images, expected_anchor = map(int, sys.argv[3:])
assert report["status"] == "complete"
assert report["scene_count"] == expected_scenes
assert report["variant_count"] == 3
assert report["png_count"] == expected_images
assert report["support_content_matches"] == expected_images
assert report["support_padding_matches"] == expected_images
assert report["reference_letterbox_matches"] == expected_anchor
print(
    json.dumps(
        {
            "status": report["status"],
            "dataset": sys.argv[2],
            "scene_count": report["scene_count"],
            "png_count": report["png_count"],
            "tree_sha256": report["tree_sha256"],
        }
    )
)
PY
