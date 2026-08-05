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

mapfile -t variants < <(
  "${python_bin}" -c \
    'import json,sys; print(*( "orbit_{}".format(x["label"]) for x in json.load(open(sys.argv[1]))["orbit"]["ordered_members"]), sep="\n")' \
    "${protocol}"
)

case "${dataset}" in
  eth3d)
    selection_root="/mnt/e/camcanon3r-data/eth3d_selected"
    selection_report="${selection_root}/selection_report.json"
    prepared_root="data/eth3d_training/raw_canonical_orbit"
    orbit_prediction_root="outputs/eth3d_training/${model}/raw_canonical_orbit"
    projection_root="outputs/eth3d_training/${model}/raw_canonical_orbit_projection"
    identity_root="outputs/eth3d_training/${model}/raw"
    analytic_root="outputs/eth3d_training/${model}/raw_canonical"
    projection_report="results/orbit/eth3d_${model}_projection.json"
    evaluation_report="results/orbit/eth3d_${model}_evaluation.json"
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
    selection_root="/mnt/e/camcanon3r-data/dtu_selected"
    prepared_root="data/dtu/rectified_canonical_orbit"
    orbit_prediction_root="outputs/dtu/${model}/rectified_canonical_orbit"
    projection_root="outputs/dtu/${model}/rectified_canonical_orbit_projection"
    identity_root="outputs/dtu/${model}/rectified_mechanism"
    analytic_root="outputs/dtu/${model}/rectified_canonical"
    projection_report="results/orbit/dtu_${model}_projection.json"
    evaluation_report="results/orbit/dtu_${model}_evaluation.json"
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
  echo "canonical orbit projection design is incomplete" >&2
  exit 1
fi

case "${model}" in
  vggt)
    PYTHONPATH=src:third_party/vggt .venv/bin/python \
      scripts/run_vggt_batch.py "${prepared_root}" "${orbit_prediction_root}" \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/VGGT-1B/model.safetensors \
      --max-views "${max_views}" --preprocess crop --seed 17 \
      --resume --audit-only >/dev/null
    ;;
  dust3r)
    PYTHONPATH=src:third_party/dust3r:third_party/dust3r/croco \
      .venv-dust3r/bin/python scripts/run_dust3r_batch.py \
      "${prepared_root}" "${orbit_prediction_root}" \
      --scenes "${scenes[@]}" --variants "${variants[@]}" \
      --weights checkpoints/dust3r-512-dpt \
      --max-views "${max_views}" --image-size 512 --batch-size 1 \
      --niter 300 --schedule cosine --lr 0.01 --seed 17 \
      --resume --audit-only >/dev/null
    ;;
esac

PYTHONPATH=src "${python_bin}" scripts/project_orbit_sweep.py \
  "${orbit_prediction_root}" "${projection_root}" --protocol "${protocol}" \
  --scenes "${scenes[@]}" --report "${projection_report}" --resume

PYTHONPATH=src "${python_bin}" scripts/evaluate_orbit_projection.py \
  "${selection_root}" "${orbit_prediction_root}" "${projection_root}" \
  "${identity_root}" "${analytic_root}" "${evaluation_report}" \
  --protocol "${protocol}" --dataset "${dataset}" \
  --dataset-label "${dataset_label}" --model "${model}" \
  --scenes "${scenes[@]}"
