#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != "vggt" && $1 != "dust3r" ) ]]; then
  echo "usage: $0 {vggt|dust3r}" >&2
  exit 2
fi

model="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
base_protocol="${repo_root}/configs/dtu_protocol.json"
repair_protocol="${repo_root}/configs/dtu_repair_protocol.json"
prepared_root="${repo_root}/data/dtu/rectified_mechanism"
repaired_root="${repo_root}/data/dtu/rectified_canonical"
repair_audit="${repo_root}/results/dtu/rectified_canonical_preparation_audit.json"
output_root="${repo_root}/outputs/dtu/${model}/rectified_canonical"

if pgrep -f "scripts/run_(vggt|dust3r)_batch.py" >/dev/null 2>&1; then
  echo "another CamCanon3R model sweep is already running" >&2
  exit 1
fi

cd "${repo_root}"
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
mapfile -t variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["ordered_repaired_variants"], sep="\n")' \
    "${repair_protocol}"
)
fill_policy="$(
  "${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["fill_policy"])' \
    "${repair_protocol}"
)"

if [[ ${#scenes[@]} -ne 22 || ${#source_variants[@]} -ne 2 || ${#variants[@]} -ne 2 ]]; then
  echo "frozen DTU repair inference must contain 22 scenes and 2 variants" >&2
  exit 1
fi

PYTHONPATH=src "${python_bin}" scripts/audit_canonical_repairs.py \
  "${prepared_root}" "${repaired_root}" \
  --scenes "${scenes[@]}" --source-variants "${source_variants[@]}" \
  --fill-policy "${fill_policy}" --output "${repair_audit}" >/dev/null

case "${model}" in
  vggt)
    PYTHONPATH=src:third_party/vggt .venv/bin/python \
      scripts/run_vggt_batch.py \
      data/dtu/rectified_canonical outputs/dtu/vggt/rectified_canonical \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/VGGT-1B/model.safetensors \
      --max-views 3 --preprocess crop --seed 17 --resume
    ;;
  dust3r)
    PYTHONPATH=src:third_party/dust3r:third_party/dust3r/croco \
      .venv-dust3r/bin/python scripts/run_dust3r_batch.py \
      data/dtu/rectified_canonical outputs/dtu/dust3r/rectified_canonical \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/dust3r-512-dpt \
      --max-views 3 --image-size 512 --batch-size 1 \
      --niter 300 --schedule cosine --lr 0.01 --seed 17 --resume
    ;;
esac

PYTHONPATH=src "${python_bin}" scripts/summarize_prediction_compute.py \
  "${output_root}" "results/dtu/${model}/repair_inference_compute.json" \
  --model "${model}" --dataset dtu-held-out-repair \
  --scenes "${scenes[@]}" --variants "${variants[@]}" \
  --require-end-to-end

PYTHONPATH=src "${python_bin}" scripts/audit_prediction_repeat.py \
  "outputs/dtu/${model}/rectified_mechanism" \
  "outputs/dtu/${model}/rectified_canonical" \
  --scenes "${scenes[@]}" --variant identity \
  --output "results/dtu/${model}/repair_identity_repeat_audit.json"
