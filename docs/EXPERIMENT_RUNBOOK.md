# my5090 experiment runbook

All commands run from `/opt/camcanon3r`. Do not start a VGGT run while another
project owns the GPU. Confirm both the process list and utilization rather than
relying on one momentary utilization sample.

Long jobs must be owned by a Windows Scheduled Task so WSL remains alive after
the SSH controller disconnects. A Linux-only `tmux` or `nohup` session is not
sufficient on this machine. From my5090 WSL, start an auditable job with one
quoted shell command:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-JOB-NAME \
  'cd /opt/camcanon3r && COMMAND > outputs/logs/JOB.log 2>&1'
```

The launcher refuses a concurrent task or a same-named task whose registered
command differs. The task has no trigger and is started only on explicit
request. Inspect its Windows state and the Linux process/log before deciding
whether a resumable job needs another start.

## ETH3D archive acquisition

The frozen full-training-scene manifest contains all 13 official scenes. Start
the resumable download through the process-scoped proxy at low CPU/I/O
priority:

```bash
./scripts/start_eth3d_download_my5090.sh
```

Progress is recorded in
`/mnt/e/camcanon3r-data/eth3d_archives/download.log`; verified byte lengths and
SHA-256 values are checkpointed after every archive in
`download_report.json`. This stage downloads only. Do not extract or prepare
full-resolution PNG variants until CPU and disk contention is safe.

After `download_report.json` has a non-null completion time and every archive
is verified, extract the frozen first four lexicographically sorted DSLR views
per scene, their raw and undistorted calibration, and matching raw depth. The
selection is independent of model outcomes and writes per-file hashes:

```bash
PYTHONPATH=src .venv/bin/python scripts/extract_eth3d_selection.py \
  configs/eth3d_training_archives.json \
  /mnt/e/camcanon3r-data/eth3d_archives \
  /mnt/e/camcanon3r-data/eth3d_selected \
  --views-per-scene 4 --resume
```

The extractor refuses an incomplete download report, validates archive member
coverage before extraction, and checks every extracted byte length before
atomically writing `selection_report.json`.

Prepare the benchmark-scale raw sweep with atomic resumable PNG writes:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_eth3d_selection.py \
  /mnt/e/camcanon3r-data/eth3d_selected data/eth3d_training/raw \
  --domain raw \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --seed 17 --resume
```

Use a separate `data/eth3d_training/undistorted` root with
`--domain undistorted` for the pose-only protocol; never mix those summaries
with raw-depth results.

After inference, evaluate the complete frozen selection in one resumable pass.
The evaluator rejects missing predictions, reordered views, stale result files,
extra evaluations outside the frozen design, and mixed raw/pose-only summaries:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_eth3d_selection.py \
  /mnt/e/camcanon3r-data/eth3d_selected \
  outputs/eth3d_training/vggt/raw results/eth3d_training/vggt/raw \
  --domain raw \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --bootstrap-replicates 10000 --bootstrap-seed 17 --resume

PYTHONPATH=src .venv/bin/python scripts/evaluate_eth3d_selection.py \
  /mnt/e/camcanon3r-data/eth3d_selected \
  outputs/eth3d_training/vggt/undistorted \
  results/eth3d_training/vggt/undistorted \
  --domain undistorted \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --bootstrap-replicates 10000 --bootstrap-seed 17 --resume
```

Each per-run result and the aggregate summary is written atomically. Resume is
allowed only when the existing record resolves to the same scene, variant,
domain, prediction, calibration, and depth source.

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
  --seed 17 --max-views 4 --resume

PYTHONPATH=src .venv/bin/python -m camcanon3r.cli prepare-scene \
  data/eth3d_office/undistorted_selected \
  data/eth3d_office/undistorted_prepared \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --seed 17 --max-views 4 --resume
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

## Analytic canonical-canvas repair

The repair uses the registered affine to inverse-warp each prepared image onto
its original camera canvas. Pixels outside visible crop support are neutral
gray and exported with a binary mask. Identity is processed through the same
path to measure clean cost.

```bash
PYTHONPATH=src .venv/bin/python scripts/canonicalize_sweep.py \
  data/pilot data/pilot_canonical \
  --scenes room kitchen llff_fern \
  --variants identity asymmetric_crop_075 asymmetric_crop_090 \
  --fill-policy neutral_gray --resume

PYTHONPATH=src:third_party/vggt .venv/bin/python scripts/run_vggt_batch.py \
  data/pilot_canonical outputs/vggt/pilot_canonical \
  --scenes room kitchen llff_fern \
  --variants identity canonical_asymmetric_crop_075 \
    canonical_asymmetric_crop_090 \
  --weights checkpoints/VGGT-1B/model.safetensors \
  --max-views 4 --preprocess crop --seed 17 --resume

PYTHONPATH=src .venv/bin/python scripts/compare_prediction_sweep.py \
  outputs/vggt/pilot_canonical results/vggt/pilot_canonical \
  --scenes room kitchen llff_fern \
  --variants canonical_asymmetric_crop_075 \
    canonical_asymmetric_crop_090 --resume
```

This diagnostic only shows whether canonicalization reduces disagreement. The
repair claim is promoted solely from paired ETH3D or DTU ground-truth gap
recovery, with the clean cost and visible-support fraction reported alongside.

After evaluating the original, corrupted, repaired, and repaired-identity
predictions against the same GT, compute each metric's raw gap and recovery:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_repair.py \
  results/eth3d_office/raw/identity.json \
  results/eth3d_office/raw/asymmetric_crop_075.json \
  results/eth3d_office/raw_canonical/canonical_asymmetric_crop_075.json \
  --clean-control results/eth3d_office/raw_canonical/identity.json \
  --output results/eth3d_office/raw_canonical/repair_gap_075.json
```

The output never clips recovery to `[0, 1]`: negative repair and better-than-
identity overshoot remain visible. A non-positive or noise-floor corruption gap
is marked undefined, and pose-only ETH3D records keep depth unavailable.

## Statistical aggregation

Sweep summaries use a deterministic 10,000-replicate percentile bootstrap
over scenes with seed 17 and 95% intervals. The same sampled scene indices are
shared across metrics so raw error and identity-relative deltas remain paired.
Summaries reject duplicate scene/variant records and reject mixtures of
pose-only and pose-plus-depth ETH3D protocols. Fewer than ten scenes trigger a
machine-readable `descriptive_only_fewer_than_10_scenes` warning.

The defaults can be reproduced or overridden explicitly:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_comparisons.py \
  results/vggt/pilot --output results/vggt/pilot/summary.json \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17
```

For a nested ETH3D layout with one directory per scene, aggregate every
`*_vs_gt.json` record while enforcing a complete paired design:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_eth3d.py \
  results/eth3d/raw --output results/eth3d/raw/summary.json \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17
```

## Reliability evaluation

After creating held-out case records with one GT error and uncertainty score
per `(model, dataset, scene, view set, transform)`, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_reliability.py \
  results/reliability/rotation_cases.jsonl \
  --failure-threshold 2.0 \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17 \
  --output results/reliability/rotation_disagreement.json
```

The exact case schema, split rules, tie handling, AURC definition, and claim
gate are frozen in `docs/RELIABILITY_PROTOCOL.md`.
