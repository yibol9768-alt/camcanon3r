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
prepared_root="${repo_root}/data/dtu/rectified_mechanism"
preparation_audit="${repo_root}/results/dtu/rectified_mechanism_preparation_audit.json"

if pgrep -f "scripts/run_(vggt|dust3r)_batch.py" >/dev/null 2>&1; then
  echo "a CamCanon3R model sweep is still running" >&2
  exit 1
fi

cd "${repo_root}"
PYTHONPATH=src "${python_bin}" scripts/audit_dtu_mechanism.py \
  "${prepared_root}" \
  --protocol "${protocol}" \
  --variant-config "${variant_config}" \
  --output "${preparation_audit}" >/dev/null

mapfile -t scenes < <(
  "${python_bin}" -c \
    'import json,sys; print(*(f"scan{value}" for value in json.load(open(sys.argv[1]))["evaluation_scans"]), sep="\n")' \
    "${protocol}"
)
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

if [[ ${#scenes[@]} -ne 22 || ${#variants[@]} -ne 11 || ${#point_variants[@]} -ne 4 ]]; then
  echo "frozen DTU evaluation must contain 11 pose and 4 point variants" >&2
  exit 1
fi

case "${model}" in
  vggt)
    PYTHONPATH=src:third_party/vggt .venv/bin/python \
      scripts/run_vggt_batch.py \
      data/dtu/rectified_mechanism outputs/dtu/vggt/rectified_mechanism \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/VGGT-1B/model.safetensors \
      --max-views 3 --preprocess crop --seed 17 --resume --audit-only >/dev/null
    ;;
  dust3r)
    PYTHONPATH=src:third_party/dust3r:third_party/dust3r/croco \
      .venv-dust3r/bin/python scripts/run_dust3r_batch.py \
      data/dtu/rectified_mechanism outputs/dtu/dust3r/rectified_mechanism \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/dust3r-512-dpt \
      --max-views 3 --image-size 512 --batch-size 1 \
      --niter 300 --schedule cosine --lr 0.01 --seed 17 \
      --resume --audit-only >/dev/null
    ;;
esac

PYTHONPATH=src "${python_bin}" scripts/evaluate_dtu_selection.py \
  /mnt/e/camcanon3r-data/dtu_selected \
  "outputs/dtu/${model}/rectified_mechanism" \
  "results/dtu/${model}/rectified_mechanism" \
  --protocol "${protocol}" \
  --variants "${variants[@]}" \
  --point-variants "${point_variants[@]}" \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17 --resume
