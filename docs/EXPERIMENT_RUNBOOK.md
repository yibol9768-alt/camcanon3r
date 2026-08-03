# my5090 experiment runbook

All commands run from `/opt/camcanon3r`. Do not start a VGGT run while another
project owns the GPU. Confirm both the process list and utilization rather than
relying on one momentary utilization sample.

## Three-scene severity sweep

The prepared inputs contain three variants for each of `room`, `kitchen`, and
`llff_fern`: `center_crop_090`, `center_crop_060`, and
`asymmetric_crop_090`.

```bash
PYTHONPATH=src:third_party/vggt .venv/bin/python scripts/run_vggt_batch.py \
  data/pilot outputs/pilot \
  --scenes room kitchen llff_fern \
  --variants center_crop_090 center_crop_060 asymmetric_crop_090 \
  --weights checkpoints/VGGT-1B/model.safetensors \
  --max-views 4 --preprocess crop --seed 17 --resume

PYTHONPATH=src .venv/bin/python scripts/compare_vggt_sweep.py \
  outputs/pilot results/pilot \
  --scenes room kitchen llff_fern \
  --variants center_crop_090 center_crop_060 asymmetric_crop_090 \
  --resume
```

The first command validates every prepared directory before loading VGGT,
loads the model once for all nine pending runs, rejects partial outputs, and
supports exact resume. The second command compares each candidate with its
scene's identity prediction and writes `results/pilot/summary.json`.

## ETH3D office preparation

The selected raw and pre-undistorted source links contain `DSC_0219` through
`DSC_0222`. Preparing the two domains is CPU and disk intensive because the
protocol preserves source resolution and losslessly writes PNG files.

```bash
PYTHONPATH=src .venv/bin/python -m camcanon3r.cli prepare-scene \
  data/eth3d_office/raw_selected data/eth3d_office/raw_prepared \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --seed 17 --max-views 4

PYTHONPATH=src .venv/bin/python -m camcanon3r.cli prepare-scene \
  data/eth3d_office/undistorted_selected \
  data/eth3d_office/undistorted_prepared \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --seed 17 --max-views 4
```

Run VGGT once per image domain:

```bash
PYTHONPATH=src:third_party/vggt .venv/bin/python scripts/run_vggt_batch.py \
  data/eth3d_office/raw_prepared outputs/eth3d_office/raw \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --weights checkpoints/VGGT-1B/model.safetensors --resume

PYTHONPATH=src:third_party/vggt .venv/bin/python scripts/run_vggt_batch.py \
  data/eth3d_office/undistorted_prepared outputs/eth3d_office/undistorted \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --weights checkpoints/VGGT-1B/model.safetensors --resume
```

Evaluate raw images with raw depth, and pre-undistorted images with pose only:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_eth3d_sweep.py \
  outputs/eth3d_office/raw \
  /mnt/e/camcanon3r-data/eth3d/office/dslr_calibration_jpg \
  results/eth3d_office/raw \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --depth-dir \
  /mnt/e/camcanon3r-data/eth3d/office/ground_truth_depth/dslr_images \
  --resume

PYTHONPATH=src .venv/bin/python scripts/evaluate_eth3d_sweep.py \
  outputs/eth3d_office/undistorted \
  /mnt/e/camcanon3r-data/eth3d/office/dslr_calibration_undistorted \
  results/eth3d_office/undistorted \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --pose-only --resume
```

Each ETH3D summary reports absolute GT metrics and the signed delta from the
identity run. Positive deltas are direct accuracy degradation; cross-run
disagreement alone must not be described as such.

## DUSt3R confirmatory matrix

DUSt3R is pinned and licensed as documented in `docs/DUST3R_PROTOCOL.md`.
Install it without changing any Windows or WSL global proxy setting:

```bash
CAMCANON3R_ACCEPT_DUST3R_NONCOMMERCIAL=1 \
  ./scripts/setup_dust3r_my5090.sh
```

Run the same three-scene diagnostic matrix with a single model load:

```bash
PYTHONPATH=src:third_party/dust3r:third_party/dust3r/croco \
  .venv-dust3r/bin/python scripts/run_dust3r_batch.py \
  data/pilot outputs/dust3r/pilot \
  --scenes room kitchen llff_fern \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --weights checkpoints/dust3r-512-dpt \
  --max-views 4 --image-size 512 --batch-size 1 \
  --niter 300 --schedule cosine --lr 0.01 --seed 17 --resume

PYTHONPATH=src .venv-dust3r/bin/python scripts/compare_prediction_sweep.py \
  outputs/dust3r/pilot results/dust3r/pilot \
  --scenes room kitchen llff_fern \
  --variants center_crop_075 asymmetric_crop_075 letterbox_square --resume
```

Do not pool VGGT and DUSt3R runs before reporting each model separately. The
same identity-relative rotation, translation-direction, and aligned-depth
metrics are used because both adapters emit the same archive schema.
