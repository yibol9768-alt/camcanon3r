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
output_dir="results/reliability/dtu_${model}_seed17"
case_file="${output_dir}/cases.json"

cd "${repo_root}"
mapfile -t variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["confirmatory_variants"], sep="\n")' \
    "${protocol}"
)
if [[ ${#variants[@]} -ne 4 ]]; then
  echo "frozen DTU reliability design must contain four variants" >&2
  exit 1
fi

PYTHONPATH=src "${python_bin}" scripts/build_reliability_cases.py \
  "outputs/dtu/${model}/rectified_mechanism" \
  "results/dtu/${model}/rectified_mechanism" \
  "${case_file}" \
  --variants "${variants[@]}" \
  --model "${model}" --dataset dtu-held-out \
  --allow-extra-variants

PYTHONPATH=src "${python_bin}" scripts/analyze_reliability.py "${case_file}" \
  --error-field ground_truth.rotation_median_degrees \
  --uncertainty-field scores.rotation_disagreement_degrees \
  --failure-threshold 2.0 --bootstrap-replicates 10000 \
  --confidence-level 0.95 --bootstrap-seed 17 \
  --output "${output_dir}/rotation_disagreement.json" >/dev/null

PYTHONPATH=src "${python_bin}" scripts/analyze_reliability.py "${case_file}" \
  --error-field ground_truth.rotation_median_degrees \
  --uncertainty-field scores.native_uncertainty \
  --failure-threshold 2.0 --bootstrap-replicates 10000 \
  --confidence-level 0.95 --bootstrap-seed 17 \
  --output "${output_dir}/rotation_native_uncertainty.json" >/dev/null

echo '{"status":"complete","dataset":"dtu-held-out"}'
