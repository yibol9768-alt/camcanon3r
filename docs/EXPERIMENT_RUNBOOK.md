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
domain, prediction, calibration, depth source, and evaluation protocol version.
Protocol-version drift is rejected rather than silently reusing legacy records.

Raw evaluation reports two separate geometry views. Scale-aligned depth AbsRel
samples the official raw z-depth at inverse-affine-mapped tensor pixels. The
point-map protocol samples and backprojects the selected raw depth through the
`THIN_PRISM_FISHEYE` model, retains predicted world points whose mapped raw
pixels have finite scan support, and fits one orientation-preserving Sim(3)
using camera poses only. Paired rotations determine the global rotation, then
camera centers determine positive scale and translation; this remains stable
for nearly collinear camera paths. Calibration ID, intrinsics, and image size
are resolved per view because a frozen scene subset may span multiple COLMAP
cameras. It then applies a 1 cm voxel grid and deterministic
100,000-point pooled cap and reports untruncated prediction-to-GT accuracy plus
GT-to-prediction completeness (mean, median, and p90 in meters). Before
pooling, raw-resolution computation is bounded by deterministically sampling at
most 100,000 finite supported pixels per view. These numbers must be described
as CamCanon3R's raw-depth-derived point-map metrics, not as official ETH3D MVS
leaderboard scores.

If a model collapses all predicted camera centers, the pose-only Sim(3) scale is
undefined. The evaluator keeps pose and depth results, emits an explicit
`undefined_degenerate_camera_center_alignment` point-map status, and excludes
that incomplete scene metric from the bootstrap instead of substituting zero.

## DTU held-out acquisition

DTU is frozen before GT inspection to the 22 standard MVSNet evaluation scans
and the canonical pixelNeRF three-view indices 22, 25, and 28 (camera IDs 23,
26, and 29 in the official one-based Rectified filenames), lighting index 3.
The exact split, reliability threshold, archive identities, and references are
in `configs/dtu_protocol.json` and `configs/dtu_sources.json`.

The official Rectified archive is about 123 GB, so do not download it in full.
After the three complete range indexes exist under
`/mnt/e/camcanon3r-data/dtu_mvs`, build exact member selections:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_dtu_remote_selections.py \
  /mnt/e/camcanon3r-data/dtu_mvs \
  /mnt/e/camcanon3r-data/dtu_mvs/selections
```

Extract each selection through the process-scoped download proxy. These are
long network jobs and must use a Windows Scheduled Task. The same output root
is intentional; the three manifests have disjoint targets:

```bash
for archive in rectified sampleset points; do
  ./scripts/with_download_proxy.sh env PYTHONPATH=src .venv/bin/python \
    scripts/extract_remote_zip_selection.py \
    /mnt/e/camcanon3r-data/dtu_mvs/selections/${archive}.json \
    /mnt/e/camcanon3r-data/dtu_selected \
    /mnt/e/camcanon3r-data/dtu_mvs/${archive}_extraction_report.json \
    --resume
done
```

Every selected member is checked against the indexed byte length and CRC-32,
written atomically, and recorded with SHA-256. The selected payload consists of
66 rectified images, 22 official point clouds, 22 observability masks, 22 ground
planes, selected calibration files, and the official evaluation code. Do not
inspect DTU GT metrics until the reliability cases, score fields, failure
threshold, and AUROC gate in `docs/RELIABILITY_PROTOCOL.md` are frozen.

Once `rectified_extraction_report.json` is complete, prepare all eleven frozen
mechanism variants. Preparation independently revalidates the exact 22 x 3
source-image design and checkpoints its protocol and extraction-report hashes:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_dtu_selection.py \
  /mnt/e/camcanon3r-data/dtu_selected \
  data/dtu/rectified_mechanism \
  /mnt/e/camcanon3r-data/dtu_mvs/rectified_extraction_report.json \
  --resume
```

The expected prepared design is 22 scene manifests plus 22 x 11 x 3 = 726
PNGs. Run preparation as a detached Windows task; do not tie it to an SSH
session.

Before inference, bind the preparation report to the frozen protocol and hash
the complete prepared image tree:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_dtu_mechanism.py \
  data/dtu/rectified_mechanism \
  --protocol configs/dtu_protocol.json \
  --variant-config configs/eth3d_mechanism_variants.json \
  --output results/dtu/rectified_mechanism_preparation_audit.json
```

The audit rejects any scene, camera, light, seed, variant, file, protocol hash,
or preparation-report drift. GPU inference starts only after it passes.

Run the two models sequentially, never concurrently, over the exact ordered
variant list in `configs/eth3d_mechanism_variants.json`. Use all 22 `scan*`
directories beneath `data/dtu/rectified_mechanism`, three views per case, and
the already verified model weights. Keep each sweep in a Windows-owned
background task and use `--resume` only after its existing NPZ/JSON pairs pass
the batch runner's validation.

After both prediction sweeps complete, evaluate each model into a separate
result root. The command below intentionally requires all eleven variants in
frozen order and exactly the four predeclared point-map variants:

```bash
variants=(
  identity center_crop_075 asymmetric_crop_075 letterbox_square
  center_crop_090 center_crop_060
  asymmetric_crop_090 asymmetric_crop_060
  shared_asymmetric_crop_090 shared_asymmetric_crop_075
  shared_asymmetric_crop_060
)
point_variants=(
  identity center_crop_075 asymmetric_crop_075 letterbox_square
)

PYTHONPATH=src .venv/bin/python scripts/evaluate_dtu_selection.py \
  /mnt/e/camcanon3r-data/dtu_selected \
  outputs/dtu/vggt/rectified_mechanism \
  results/dtu/vggt/rectified_mechanism \
  --protocol configs/dtu_protocol.json \
  --variants "${variants[@]}" \
  --point-variants "${point_variants[@]}" \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17 --resume
```

Repeat for DUSt3R by changing only the model-specific prediction and result
roots. The evaluator verifies every source view, calibration file, prediction
pair, GT resource, protocol hash, and variant-config hash before computing any
metric. Pose and intrinsics are evaluated for all 242 cases per model; point
accuracy and completeness are evaluated only for the frozen 88 confirmatory
cases. Its surface metric adopts the DTU observability mask, ground-plane
filter, 0.2 mm voxel spacing, and 20 mm distance rejection, but uses a
deterministic 100,000-point cap and camera-pose-only Sim(3) gauge alignment.
Therefore report it as CamCanon3R's deterministic DTU point-map metric, not an
official DTU leaderboard score.

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

## Benchmark-scale severity and crop-scope matrix

The pre-result design is frozen in
`configs/eth3d_mechanism_variants.json`.  Preserve its exact variant order:
the first four entries intentionally reproduce the completed seed-17
confirmatory transforms, and the appended entries receive fixed seeds via the
10,007 stride.  Reordering the list changes stochastic crop windows and is a
different experiment.

```bash
variants=(
  identity center_crop_075 asymmetric_crop_075 letterbox_square
  center_crop_090 center_crop_060
  asymmetric_crop_090 asymmetric_crop_060
  shared_asymmetric_crop_090 shared_asymmetric_crop_075
  shared_asymmetric_crop_060
)

PYTHONPATH=src .venv/bin/python scripts/prepare_eth3d_selection.py \
  /mnt/e/camcanon3r-data/eth3d_selected \
  data/eth3d_training/raw_mechanism --domain raw \
  --variants "${variants[@]}" --seed 17 --resume
```

Run VGGT and DUSt3R in separate Windows-owned background tasks using the same
ordered array, then evaluate each model separately with
`scripts/evaluate_eth3d_selection.py`.  Never pool the model summaries.  The
mechanism contrasts are:

- retained fraction 90%, 75%, and 60% within center, shared off-center, and
  independently shifted crop families;
- shared versus independent off-center windows at matched retained fraction;
- identity and letterbox negative controls.

`shared_asymmetric_crop_*` resets the registered RNG for every view, so its
normalized crop window is identical across the set even if source resolutions
differ.  `asymmetric_crop_*` advances the RNG and remains view dependent.  A
second transform family passes the frozen hypothesis threshold only if its
paired rotation degradation exceeds 2 degrees or its depth AbsRel increase
exceeds 0.05 on both registered datasets; a significant but smaller effect is
reported without moving the threshold.

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

PYTHONPATH=src .venv/bin/python scripts/audit_canonical_repairs.py \
  data/eth3d_training/raw data/eth3d_training/raw_canonical \
  --scenes courtyard delivery_area electro facade kicker meadow office \
    pipes playground relief relief_2 terrace terrains \
  --source-variants identity asymmetric_crop_075 \
  --output results/repair/eth3d_raw_canonical_preparation_audit.json

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

The audit must pass before GPU inference. It checks the complete scene/variant
file design, every image and binary validity mask, source-affine provenance,
identity pixel equality, fill pixels outside valid support, per-image valid
fractions, and a deterministic tree hash.

Before any repair GT result is inspected, prepare and audit the two additional
frozen fill policies in separate roots. Include identity in each root so the
standard GT evaluator retains a complete paired design:

```bash
PYTHONPATH=src .venv/bin/python scripts/canonicalize_sweep.py \
  data/eth3d_training/raw data/eth3d_training/raw_canonical_black \
  --scenes courtyard delivery_area electro facade kicker meadow office \
    pipes playground relief relief_2 terrace terrains \
  --variants identity asymmetric_crop_075 \
  --prefix canonical_black_ --fill-policy black --resume

PYTHONPATH=src .venv/bin/python scripts/canonicalize_sweep.py \
  data/eth3d_training/raw data/eth3d_training/raw_canonical_mean \
  --scenes courtyard delivery_area electro facade kicker meadow office \
    pipes playground relief relief_2 terrace terrains \
  --variants identity asymmetric_crop_075 \
  --prefix canonical_mean_ --fill-policy image_mean --resume
```

Audit each root with `scripts/audit_canonical_repairs.py`, using the matching
prefix and the two source variants. Run VGGT and DUSt3R sequentially on each
root, then evaluate each root against ETH3D GT. The three repaired candidate
names and order must exactly match `configs/repair_consensus_protocol.json`.
Finally run the matched selection analysis, separately per model:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_consensus_repair.py \
  results/eth3d_training/vggt/raw \
  results/eth3d_training/vggt/raw_canonical \
  results/repair/eth3d_vggt_raw_consensus.json \
  --protocol configs/repair_consensus_protocol.json \
  --candidate neutral_gray canonical_asymmetric_crop_075 \
    outputs/eth3d_training/vggt/raw_canonical \
    results/eth3d_training/vggt/raw_canonical \
  --candidate black canonical_black_asymmetric_crop_075 \
    outputs/eth3d_training/vggt/raw_canonical_black \
    results/eth3d_training/vggt/raw_canonical_black \
  --candidate image_mean canonical_mean_asymmetric_crop_075 \
    outputs/eth3d_training/vggt/raw_canonical_mean \
    results/eth3d_training/vggt/raw_canonical_mean \
  --model vggt --dataset eth3d-training-raw \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17
```

Repeat for DUSt3R by changing all model-specific roots and `--model`. This
single report compares neutral-gray analytic repair, cross-fill consensus,
native-confidence selection, and the GT-rotation oracle. Consensus and the two
matched baselines all consume the same three predictions; selection frequency,
runtime, peak VRAM, and recovery per model-compute second remain explicit.

This diagnostic only shows whether canonicalization reduces disagreement. The
repair claim is promoted solely from paired ETH3D or DTU ground-truth gap
recovery, with the clean cost and visible-support fraction reported alongside.

After evaluating the original, corrupted, repaired, and repaired-identity
predictions against the same GT, aggregate the complete paired scene design:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_repair_selection.py \
  results/eth3d_training/vggt/raw \
  results/eth3d_training/vggt/raw_canonical \
  results/repair/eth3d_vggt_raw_asymmetric_crop_075.json \
  --identity-variant identity \
  --corrupt-variant asymmetric_crop_075 \
  --clean-control-variant identity \
  --repaired-variant canonical_asymmetric_crop_075 \
  --model vggt --dataset eth3d-training-raw \
  --recovery-threshold 0.30 --clean-relative-threshold 0.02 \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17
```

Repeat without changing thresholds for DUSt3R and later DTU.  Aggregate gap
recovery is the median paired recovered gap divided by the median paired
corruption gap; bootstrap replicates resample scenes and recompute that ratio.
The output never clips recovery to `[0, 1]`: negative repair and better-than-
identity overshoot remain visible. A non-positive or noise-floor corruption
gap is marked undefined. The registered point-estimate gate is at least 30%
recovery and at most 2% median relative clean degradation; confidence-bound
versions are reported separately and never silently substituted for the
registered point-estimate rule. Pose-only records keep depth unavailable.

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

Build complete case records before evaluation. ETH3D is development-only for
this detector because its outcomes were already inspected when the score was
designed:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_reliability_cases.py \
  outputs/eth3d_training/vggt/raw \
  results/eth3d_training/vggt/raw \
  results/reliability/eth3d_vggt_raw_seed17/cases.json \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --model vggt --dataset eth3d-training-raw \
  --allow-extra-variants
```

Use `--allow-extra-variants` only after the complete eleven-variant source
sweep has passed its exact-design audit. The builder still constructs scores
and cases from exactly the four listed registered candidates and records that
the enclosing roots contained an audited superset.

Then run one score/error pair at a time using nested fields:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_reliability.py \
  results/reliability/eth3d_vggt_raw_seed17/cases.json \
  --error-field ground_truth.rotation_median_degrees \
  --uncertainty-field scores.rotation_disagreement_degrees \
  --failure-threshold 2.0 \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17 \
  --output results/reliability/rotation_disagreement.json
```

Repeat with `scores.native_uncertainty` as the required native-confidence
baseline. The primary held-out endpoint is rotation error above 2 degrees;
depth AbsRel above 0.05 is secondary where depth GT exists. The exact case
schema, DTU held-out status, split rules, tie handling, AURC definition, and
claim gate are frozen in `docs/RELIABILITY_PROTOCOL.md`.
