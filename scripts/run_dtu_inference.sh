#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != "vggt" && $1 != "dust3r" ) ]]; then
  echo "usage: $0 {vggt|dust3r}" >&2
  exit 2
fi

model="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prepared_root="${repo_root}/data/dtu/rectified_mechanism"
protocol="${repo_root}/configs/dtu_protocol.json"
variant_config="${repo_root}/configs/eth3d_mechanism_variants.json"
audit_output="${repo_root}/results/dtu/rectified_mechanism_preparation_audit.json"
base_python="${repo_root}/.venv/bin/python"

if pgrep -f "scripts/run_(vggt|dust3r)_batch.py" >/dev/null 2>&1; then
  echo "another CamCanon3R model sweep is already running" >&2
  exit 1
fi

cd "${repo_root}"
PYTHONPATH=src "${base_python}" scripts/audit_dtu_mechanism.py \
  "${prepared_root}" \
  --protocol "${protocol}" \
  --variant-config "${variant_config}" \
  --output "${audit_output}" >/dev/null

mapfile -t scenes < <(
  "${base_python}" -c \
    'import json,sys; print(*("scan%s" % value for value in json.load(open(sys.argv[1]))["evaluation_scans"]), sep="\n")' \
    "${protocol}"
)
mapfile -t variants < <(
  "${base_python}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["ordered_variants"], sep="\n")' \
    "${variant_config}"
)

if [[ ${#scenes[@]} -ne 22 || ${#variants[@]} -ne 11 ]]; then
  echo "frozen DTU design must contain 22 scenes and 11 variants" >&2
  exit 1
fi

case "${model}" in
  vggt)
    PYTHONPATH=src:third_party/vggt .venv/bin/python \
      scripts/run_vggt_batch.py \
      data/dtu/rectified_mechanism outputs/dtu/vggt/rectified_mechanism \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/VGGT-1B/model.safetensors \
      --max-views 3 --preprocess crop --seed 17 --resume
    ;;
  dust3r)
    PYTHONPATH=src:third_party/dust3r:third_party/dust3r/croco \
      .venv-dust3r/bin/python scripts/run_dust3r_batch.py \
      data/dtu/rectified_mechanism outputs/dtu/dust3r/rectified_mechanism \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/dust3r-512-dpt \
      --max-views 3 --image-size 512 --batch-size 1 \
      --niter 300 --schedule cosine --lr 0.01 --seed 17 --resume
    ;;
esac

PYTHONPATH=src "${base_python}" scripts/summarize_prediction_compute.py \
  "outputs/dtu/${model}/rectified_mechanism" \
  "results/dtu/${model}/inference_compute.json" \
  --model "${model}" --dataset dtu-held-out \
  --scenes "${scenes[@]}" --variants "${variants[@]}" \
  --require-end-to-end
