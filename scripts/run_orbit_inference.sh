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
protocol="${repo_root}/configs/orbit_projection_protocol.json"
cd "${repo_root}"

if pgrep -f "scripts/run_(vggt|dust3r)_batch.py" >/dev/null 2>&1; then
  echo "another CamCanon3R model sweep is already running" >&2
  exit 1
fi

mapfile -t variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*( "orbit_{}".format(x["label"]) for x in json.load(open(sys.argv[1]))["orbit"]["ordered_members"]), sep="\n")' \
    "${protocol}"
)

case "${dataset}" in
  eth3d)
    selection_report="/mnt/e/camcanon3r-data/eth3d_selected/selection_report.json"
    canonical_root="data/eth3d_training/raw_canonical"
    prepared_root="data/eth3d_training/raw_canonical_orbit"
    output_root="outputs/eth3d_training/${model}/raw_canonical_orbit"
    audit_report="results/orbit/eth3d_preparation_audit.json"
    compute_report="results/orbit/eth3d_${model}_inference_compute.json"
    dataset_label="eth3d-training-raw-canonical-orbit"
    max_views=4
    expected_scenes=13
    mapfile -t scenes < <(
      "${python_bin}" -c \
        'import json,sys; print(*(x["scene"] for x in json.load(open(sys.argv[1]))["selection"]["scenes"]), sep="\n")' \
        "${selection_report}"
    )
    ;;
  dtu)
    dtu_protocol="configs/dtu_protocol.json"
    canonical_root="data/dtu/rectified_canonical"
    prepared_root="data/dtu/rectified_canonical_orbit"
    output_root="outputs/dtu/${model}/rectified_canonical_orbit"
    audit_report="results/orbit/dtu_preparation_audit.json"
    compute_report="results/orbit/dtu_${model}_inference_compute.json"
    dataset_label="dtu-canonical-orbit"
    max_views=3
    expected_scenes=22
    mapfile -t scenes < <(
      "${python_bin}" -c \
        'import json,sys; print(*(f"scan{x}" for x in json.load(open(sys.argv[1]))["evaluation_scans"]), sep="\n")' \
        "${dtu_protocol}"
    )
    ;;
esac

if [[ ${#variants[@]} -ne 9 || ${#scenes[@]} -ne ${expected_scenes} ]]; then
  echo "canonical orbit inference design is incomplete" >&2
  exit 1
fi

PYTHONPATH=src "${python_bin}" scripts/audit_orbit_sweep.py \
  "${canonical_root}" "${prepared_root}" --protocol "${protocol}" \
  --scenes "${scenes[@]}" --output "${audit_report}" >/dev/null

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
  "${output_root}" "${compute_report}" --model "${model}" \
  --dataset "${dataset_label}" --scenes "${scenes[@]}" \
  --variants "${variants[@]}" --require-end-to-end
