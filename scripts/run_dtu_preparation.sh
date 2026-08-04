#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
archive_root="/mnt/e/camcanon3r-data/dtu_mvs"
selection_root="/mnt/e/camcanon3r-data/dtu_selected"
prepared_root="${repo_root}/data/dtu/rectified_mechanism"
audit_output="${repo_root}/results/dtu/rectified_mechanism_preparation_audit.json"

cd "${repo_root}"
"${python_bin}" - "${archive_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("sampleset", "rectified", "points"):
    path = root / f"{name}_extraction_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"DTU extraction report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete"
        or report.get("completed_members") != report.get("member_count")
    ):
        raise RuntimeError(f"DTU extraction is incomplete: {path}")
PY

PYTHONPATH=src "${python_bin}" scripts/prepare_dtu_selection.py \
  "${selection_root}" "${prepared_root}" \
  "${archive_root}/rectified_extraction_report.json" --resume

PYTHONPATH=src "${python_bin}" scripts/audit_dtu_mechanism.py \
  "${prepared_root}" \
  --protocol configs/dtu_protocol.json \
  --variant-config configs/eth3d_mechanism_variants.json \
  --output "${audit_output}" >/dev/null

"${python_bin}" - "${audit_output}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "complete"
assert report["scene_count"] == 22
assert report["variant_count"] == 11
assert report["png_count"] == 726
print(
    json.dumps(
        {
            "status": report["status"],
            "scene_count": report["scene_count"],
            "variant_count": report["variant_count"],
            "png_count": report["png_count"],
            "tree_sha256": report["tree_sha256"],
        }
    )
)
PY
