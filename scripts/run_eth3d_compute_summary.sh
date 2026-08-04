#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != "vggt" && $1 != "dust3r" ) ]]; then
  echo "usage: $0 {vggt|dust3r}" >&2
  exit 2
fi

model="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
selection_report="/mnt/e/camcanon3r-data/eth3d_selected/selection_report.json"
variant_config="${repo_root}/configs/eth3d_mechanism_variants.json"

cd "${repo_root}"
mapfile -t scenes < <(
  "${python_bin}" -c \
    'import json,sys; print(*(record["scene"] for record in json.load(open(sys.argv[1]))["selection"]["scenes"]), sep="\n")' \
    "${selection_report}"
)
mapfile -t variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["ordered_variants"], sep="\n")' \
    "${variant_config}"
)
if [[ ${#scenes[@]} -ne 13 || ${#variants[@]} -ne 11 ]]; then
  echo "frozen ETH3D compute design must contain 13 scenes and 11 variants" >&2
  exit 1
fi

PYTHONPATH=src "${python_bin}" scripts/summarize_prediction_compute.py \
  "outputs/eth3d_training/${model}/raw_mechanism" \
  "results/eth3d_training/${model}/inference_compute.json" \
  --model "${model}" --dataset eth3d-training-raw \
  --scenes "${scenes[@]}" --variants "${variants[@]}"
