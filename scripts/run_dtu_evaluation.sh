#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != "vggt" && $1 != "dust3r" ) ]]; then
  echo "usage: $0 {vggt|dust3r}" >&2
  exit 2
fi

model="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
protocol="${repo_root}/configs/dtu_protocol.json"
variant_config="${repo_root}/configs/eth3d_mechanism_variants.json"

cd "${repo_root}"
mapfile -t variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["ordered_variants"], sep="\n")' \
    "${variant_config}"
)
mapfile -t point_variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["confirmatory_variants"], sep="\n")' \
    "${protocol}"
)

if [[ ${#variants[@]} -ne 11 || ${#point_variants[@]} -ne 4 ]]; then
  echo "frozen DTU evaluation must contain 11 pose and 4 point variants" >&2
  exit 1
fi

PYTHONPATH=src "${python_bin}" scripts/evaluate_dtu_selection.py \
  /mnt/e/camcanon3r-data/dtu_selected \
  "outputs/dtu/${model}/rectified_mechanism" \
  "results/dtu/${model}/rectified_mechanism" \
  --protocol "${protocol}" \
  --variants "${variants[@]}" \
  --point-variants "${point_variants[@]}" \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17 --resume
