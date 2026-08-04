#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
base_protocol="${repo_root}/configs/dtu_protocol.json"
repair_protocol="${repo_root}/configs/dtu_repair_protocol.json"
variant_config="${repo_root}/configs/eth3d_mechanism_variants.json"
prepared_root="${repo_root}/data/dtu/rectified_mechanism"
repaired_root="${repo_root}/data/dtu/rectified_canonical"
mechanism_audit="${repo_root}/results/dtu/rectified_mechanism_preparation_audit.json"
repair_audit="${repo_root}/results/dtu/rectified_canonical_preparation_audit.json"

cd "${repo_root}"
PYTHONPATH=src "${python_bin}" scripts/audit_dtu_mechanism.py \
  "${prepared_root}" \
  --protocol "${base_protocol}" \
  --variant-config "${variant_config}" \
  --output "${mechanism_audit}" >/dev/null

mapfile -t scenes < <(
  "${python_bin}" -c \
    'import json,sys; print(*(f"scan{value}" for value in json.load(open(sys.argv[1]))["evaluation_scans"]), sep="\n")' \
    "${base_protocol}"
)
mapfile -t source_variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["source_variants"], sep="\n")' \
    "${repair_protocol}"
)
fill_policy="$(
  "${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["fill_policy"])' \
    "${repair_protocol}"
)"

if [[ ${#scenes[@]} -ne 22 || ${#source_variants[@]} -ne 2 ]]; then
  echo "frozen DTU repair preparation must contain 22 scenes and 2 variants" >&2
  exit 1
fi

PYTHONPATH=src "${python_bin}" scripts/canonicalize_sweep.py \
  "${prepared_root}" "${repaired_root}" \
  --scenes "${scenes[@]}" --variants "${source_variants[@]}" \
  --fill-policy "${fill_policy}" --resume

PYTHONPATH=src "${python_bin}" scripts/audit_canonical_repairs.py \
  "${prepared_root}" "${repaired_root}" \
  --scenes "${scenes[@]}" --source-variants "${source_variants[@]}" \
  --fill-policy "${fill_policy}" --output "${repair_audit}" >/dev/null

"${python_bin}" - "${repair_audit}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "complete"
assert report["scene_count"] == 22
assert report["source_variants"] == ["identity", "asymmetric_crop_075"]
assert report["output_variants"] == [
    "identity",
    "canonical_asymmetric_crop_075",
]
assert report["fill_policy"] == "neutral_gray"
assert report["image_count"] == 132
assert report["mask_count"] == 132
assert report["identity_pixel_matches"] == 66
print(
    json.dumps(
        {
            "status": report["status"],
            "scene_count": report["scene_count"],
            "image_count": report["image_count"],
            "mask_count": report["mask_count"],
            "tree_sha256": report["tree_sha256"],
        }
    )
)
PY
