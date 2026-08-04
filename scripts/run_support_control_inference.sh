#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( $1 != "vggt" && $1 != "dust3r" ) || ( $2 != "eth3d" && $2 != "dtu" ) ]]; then
  echo "usage: $0 {vggt|dust3r} {eth3d|dtu}" >&2
  exit 2
fi

model="$1"
dataset="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
variant_config="${repo_root}/configs/support_control_variants.json"

if pgrep -f "scripts/run_(vggt|dust3r)_batch.py" >/dev/null 2>&1; then
  echo "another CamCanon3R model sweep is already running" >&2
  exit 1
fi

cd "${repo_root}"
mapfile -t variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["ordered_variants"], sep="\n")' \
    "${variant_config}"
)

case "${dataset}" in
  eth3d)
    selection_report="/mnt/e/camcanon3r-data/eth3d_selected/selection_report.json"
    prepared_root="data/eth3d_training/raw_support_control"
    reference_root="data/eth3d_training/raw_mechanism"
    output_root="outputs/eth3d_training/${model}/raw_support_control"
    compute_output="results/eth3d_training/${model}/support_control_inference_compute.json"
    dataset_label="eth3d-training-raw-support-control"
    max_views=4
    mapfile -t scenes < <(
      "${python_bin}" -c \
        'import json,sys; print(*(x["scene"] for x in json.load(open(sys.argv[1]))["selection"]["scenes"]), sep="\n")' \
        "${selection_report}"
    )
    PYTHONPATH=src "${python_bin}" scripts/audit_support_control.py \
      "${prepared_root}" "${reference_root}" \
      --variant-config "${variant_config}" \
      --eth3d-selection-report "${selection_report}" \
      --output results/eth3d_training/support_control_preparation_audit.json \
      >/dev/null
    ;;
  dtu)
    protocol="configs/dtu_support_control_protocol.json"
    prepared_root="data/dtu/rectified_support_control"
    reference_root="data/dtu/rectified_mechanism"
    output_root="outputs/dtu/${model}/rectified_support_control"
    compute_output="results/dtu/${model}/support_control_inference_compute.json"
    dataset_label="dtu-held-out-support-control"
    max_views=3
    mapfile -t scenes < <(
      "${python_bin}" -c \
        'import json,sys; print(*(f"scan{x}" for x in json.load(open(sys.argv[1]))["evaluation_scans"]), sep="\n")' \
        "${protocol}"
    )
    PYTHONPATH=src "${python_bin}" scripts/audit_dtu_mechanism.py \
      "${prepared_root}" --protocol "${protocol}" \
      --variant-config "${variant_config}" \
      --output results/dtu/support_control_preparation_audit.json >/dev/null
    PYTHONPATH=src "${python_bin}" scripts/audit_support_control.py \
      "${prepared_root}" "${reference_root}" \
      --variant-config "${variant_config}" --dtu-protocol "${protocol}" \
      --output results/dtu/support_control_content_audit.json >/dev/null
    ;;
esac

if [[ ${#variants[@]} -ne 3 || ( ${dataset} == "eth3d" && ${#scenes[@]} -ne 13 ) || ( ${dataset} == "dtu" && ${#scenes[@]} -ne 22 ) ]]; then
  echo "support-control inference design is incomplete" >&2
  exit 1
fi

case "${model}" in
  vggt)
    PYTHONPATH=src:third_party/vggt .venv/bin/python \
      scripts/run_vggt_batch.py "${prepared_root}" "${output_root}" \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/VGGT-1B/model.safetensors \
      --max-views "${max_views}" --preprocess crop --seed 17 --resume
    ;;
  dust3r)
    PYTHONPATH=src:third_party/dust3r:third_party/dust3r/croco \
      .venv-dust3r/bin/python scripts/run_dust3r_batch.py \
      "${prepared_root}" "${output_root}" \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/dust3r-512-dpt \
      --max-views "${max_views}" --image-size 512 --batch-size 1 \
      --niter 300 --schedule cosine --lr 0.01 --seed 17 --resume
    ;;
esac

PYTHONPATH=src "${python_bin}" scripts/summarize_prediction_compute.py \
  "${output_root}" "${compute_output}" \
  --model "${model}" --dataset "${dataset_label}" \
  --scenes "${scenes[@]}" --variants "${variants[@]}" --require-end-to-end
