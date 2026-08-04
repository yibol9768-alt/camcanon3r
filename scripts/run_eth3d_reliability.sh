#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [EXECUTION_REPOSITORY]" >&2
  exit 2
fi

tool_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
execution_root="${1:-$tool_root}"
python_bin="${execution_root}/.venv/bin/python"
variants=(identity center_crop_075 asymmetric_crop_075 letterbox_square)

if [[ ! -x "${python_bin}" ]]; then
  echo "experiment Python is unavailable: ${python_bin}" >&2
  exit 1
fi

cd "${execution_root}"
for model in vggt dust3r; do
  case_file="results/reliability/eth3d_${model}_raw_seed17/cases.json"
  output_dir="$(dirname "${case_file}")"
  PYTHONPATH="${tool_root}/src" "${python_bin}" \
    "${tool_root}/scripts/build_reliability_cases.py" \
    "outputs/eth3d_training/${model}/raw_mechanism" \
    "results/eth3d_training/${model}/raw_mechanism" \
    "${case_file}" \
    --variants "${variants[@]}" \
    --model "${model}" --dataset eth3d-training-raw \
    --allow-extra-variants

  PYTHONPATH="${tool_root}/src" "${python_bin}" \
    "${tool_root}/scripts/analyze_reliability.py" "${case_file}" \
    --error-field ground_truth.rotation_median_degrees \
    --uncertainty-field scores.rotation_disagreement_degrees \
    --failure-threshold 2.0 --bootstrap-replicates 10000 \
    --confidence-level 0.95 --bootstrap-seed 17 \
    --output "${output_dir}/rotation_disagreement.json" >/dev/null
  PYTHONPATH="${tool_root}/src" "${python_bin}" \
    "${tool_root}/scripts/analyze_reliability.py" "${case_file}" \
    --error-field ground_truth.rotation_median_degrees \
    --uncertainty-field scores.native_uncertainty \
    --failure-threshold 2.0 --bootstrap-replicates 10000 \
    --confidence-level 0.95 --bootstrap-seed 17 \
    --output "${output_dir}/rotation_native_uncertainty.json" >/dev/null
  PYTHONPATH="${tool_root}/src" "${python_bin}" \
    "${tool_root}/scripts/analyze_reliability.py" "${case_file}" \
    --error-field ground_truth.depth_mean_abs_rel \
    --uncertainty-field scores.depth_disagreement_abs_rel \
    --failure-threshold 0.05 --bootstrap-replicates 10000 \
    --confidence-level 0.95 --bootstrap-seed 17 \
    --output "${output_dir}/depth_disagreement.json" >/dev/null
  PYTHONPATH="${tool_root}/src" "${python_bin}" \
    "${tool_root}/scripts/analyze_reliability.py" "${case_file}" \
    --error-field ground_truth.depth_mean_abs_rel \
    --uncertainty-field scores.native_uncertainty \
    --failure-threshold 0.05 --bootstrap-replicates 10000 \
    --confidence-level 0.95 --bootstrap-seed 17 \
    --output "${output_dir}/depth_native_uncertainty.json" >/dev/null
done

echo '{"status":"complete","models":["vggt","dust3r"],"dataset":"eth3d-training-raw"}'
