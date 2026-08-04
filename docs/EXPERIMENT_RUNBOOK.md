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

Once all three extraction reports are complete, prepare all eleven frozen
mechanism variants. Preparation independently revalidates the exact 22 x 3
source-image design and checkpoints its protocol and extraction-report hashes.
The wrapper refuses incomplete selection reports, prepares resumably, runs the
strict tree audit, and requires exactly 22 scenes, 11 variants, and 726 PNGs:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-DTUPreparation \
  'cd /opt/camcanon3r; ./scripts/run_dtu_preparation.sh \
  > results/dtu/preparation.log 2>&1'
```

The expected prepared design is 22 scene manifests plus 22 x 11 x 3 = 726
PNGs. The audit rejects any scene, camera, light, seed, variant, file, protocol
hash, or preparation-report drift. GPU inference starts only after the
preparation task is Ready with exit code 0.

Run the two models sequentially, never concurrently, over the exact ordered
variant list in `configs/eth3d_mechanism_variants.json`. Use all 22 `scan*`
directories beneath `data/dtu/rectified_mechanism`, three views per case, and
the already verified model weights. Keep each sweep in a Windows-owned
background task and use `--resume` only after its existing NPZ/JSON pairs pass
the batch runner's validation.

The model-neutral launcher re-runs the strict preparation audit, reads the 22
scenes and 11 variants directly from the frozen JSON files, refuses concurrent
CamCanon3R model inference, and selects the correct model environment.  A
resumed run does not trust filename existence alone: every skipped NPZ/JSON
pair must pass archive CRC, required-array, view-order, weight, seed, and affine
composition checks.  New schema-1.2 predictions additionally bind the ordered
prepared inputs by SHA-256, so a same-name input replacement cannot silently
reuse stale GPU output. Each new record separately times one-time model load,
model compute, and the per-scene end-to-end path through compressed NPZ write,
and records peak VRAM. On successful completion the wrapper validates the
exact metadata design and writes `results/dtu/<model>/inference_compute.json`:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-DTUVGGT \
  'cd /opt/camcanon3r; ./scripts/run_dtu_inference.sh vggt \
  > results/dtu/vggt_inference.log 2>&1'

# Start only after the VGGT task is Ready with exit code 0.
./scripts/start_my5090_background_job.sh CamCanon3R-DTUDUSt3R \
  'cd /opt/camcanon3r; ./scripts/run_dtu_inference.sh dust3r \
  > results/dtu/dust3r_inference.log 2>&1'
```

After both prediction sweeps complete, evaluate each model into a separate
result root. Before reading a prediction, the evaluation wrapper re-audits the
prepared tree and invokes the corresponding model runner in `--audit-only`
mode over all 242 NPZ/JSON pairs.  This mode refuses incomplete output rather
than launching missing inference. The frozen wrapper then requires all eleven
variants in config order and exactly the four predeclared point-map variants:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-DTUVGGTEval \
  'cd /opt/camcanon3r; ./scripts/run_dtu_evaluation.sh vggt \
  > results/dtu/vggt_evaluation.log 2>&1'

# Start only after the VGGT evaluation task is Ready with exit code 0.
./scripts/start_my5090_background_job.sh CamCanon3R-DTUDUSt3REval \
  'cd /opt/camcanon3r; ./scripts/run_dtu_evaluation.sh dust3r \
  > results/dtu/dust3r_evaluation.log 2>&1'
```

The evaluator verifies every source view, calibration file, prediction pair,
GT resource, protocol hash, and variant-config hash before computing any
metric. Pose and intrinsics are evaluated for all 242 cases per model; point
accuracy and completeness are evaluated only for the frozen 88 confirmatory
cases. Matching the official directionality, the DTU observation mask filters
prediction-to-GT accuracy only, while the ground plane filters GT-to-prediction
completeness only. The surface metric otherwise adopts 0.2 mm voxel spacing
and 20 mm distance rejection, but uses a deterministic 100,000-point cap and
camera-pose-only Sim(3) gauge alignment.
Therefore report it as CamCanon3R's deterministic DTU point-map metric, not an
official DTU leaderboard score.

If predicted camera centers collapse or imply a non-positive pose-only scale,
the evaluator preserves pose and intrinsics, marks DTU point accuracy and
completeness explicitly undefined, and excludes only those incomplete scene
metrics from bootstrap intervals. It never substitutes zero or drops the
scene from the other metrics.

Only after both GT summaries are complete, open the held-out detector result
with the unchanged four variants, score fields, strict 2-degree threshold,
10,000 scene-cluster bootstrap replicates, and seed 17:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-DTUVGGTReliability \
  'cd /opt/camcanon3r; ./scripts/run_dtu_reliability.sh vggt \
  > results/reliability/dtu_vggt_seed17.log 2>&1'

# Start only after the VGGT reliability task is Ready with exit code 0.
./scripts/start_my5090_background_job.sh CamCanon3R-DTUDUSt3RReliability \
  'cd /opt/camcanon3r; ./scripts/run_dtu_reliability.sh dust3r \
  > results/reliability/dtu_dust3r_seed17.log 2>&1'
```

The frozen qualitative protocol and cross-dataset repair claim also require a
DTU canonical-control sweep.  This is deliberately separate from the eleven-
variant mechanism sweep.  Its registered design is neutral-gray fill over
`identity` and `asymmetric_crop_075`, producing exactly 22 x 2 x 3 = 132
images plus 132 validity masks.  Preparation re-audits the main DTU input tree,
enforces the fill policy, verifies identity pixels exactly, and records a tree
hash.  It also atomically checkpoints inverse-warp wall time after every
scene/variant; a resumed output without its timing record is rejected rather
than reported as complete compute:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-DTURepairPreparation \
  'cd /opt/camcanon3r; ./scripts/run_dtu_repair_preparation.sh \
  > results/dtu/repair_preparation.log 2>&1'
```

After the preparation task exits successfully, run the two models
sequentially.  Each model executes 44 predictions, writes a separate repair
compute report, and compares every canonical-identity prediction array with
the matching main-sweep identity array.  Numerical repeat drift is retained in
the audit rather than assumed away:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-DTURepairVGGT \
  'cd /opt/camcanon3r; ./scripts/run_dtu_repair_inference.sh vggt \
  > results/dtu/vggt_repair_inference.log 2>&1'

# Start only after the VGGT repair task is Ready with exit code 0.
./scripts/start_my5090_background_job.sh CamCanon3R-DTURepairDUSt3R \
  'cd /opt/camcanon3r; ./scripts/run_dtu_repair_inference.sh dust3r \
  > results/dtu/dust3r_repair_inference.log 2>&1'
```

Do not inspect repair GT results until both main DTU summaries and both frozen
held-out reliability reports above are complete.  Then evaluate both repaired
variants with camera, intrinsics, and deterministic point-map metrics on all 22
scenes.  The evaluator binds each result to the repair protocol, base DTU
protocol, qualitative protocol, preparation-audit file, and audited input-tree
hash.  The wrapper finally compares main identity/crop against repaired
identity/canonical crop with the unchanged 30% recovery and 2% clean-cost
gates:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-DTURepairVGGTEval \
  'cd /opt/camcanon3r; ./scripts/run_dtu_repair_evaluation.sh vggt \
  > results/dtu/vggt_repair_evaluation.log 2>&1'

# Start only after the VGGT repair evaluation is Ready with exit code 0.
./scripts/start_my5090_background_job.sh CamCanon3R-DTURepairDUSt3REval \
  'cd /opt/camcanon3r; ./scripts/run_dtu_repair_evaluation.sh dust3r \
  > results/dtu/dust3r_repair_evaluation.log 2>&1'
```

The paired reports are written to
`results/repair/dtu_<model>_neutral_gray.json`.  DTU repair outcomes cannot
replace the already frozen ETH3D fill or consensus choice; they only confirm
or reject transfer of the one-pass neutral-gray orientation repair.

After all repair evaluations are frozen, render the primary qualitative grids
from the four outcome-independent scenes in `configs/qualitative_protocol.json`.
The renderer refuses model, scene, variant, prediction-path, or evaluation-
input drift.  It applies the evaluator's recorded camera-pose-only Sim(3),
projects into the first frozen target camera, normalizes depth only by the
target-camera baseline, uses one fixed raster/color range, and places the
canonical validity mask inside every repaired panel.  It retains undefined
geometry as a labeled panel.  Run it on `my5090`, where the full predictions
and calibration files reside:

```bash
MPLBACKEND=Agg PYTHONPATH=src python3 scripts/render_qualitative_grid.py \
  eth3d-training-raw results/paper/qualitative_eth3d.pdf \
  --model vggt \
    outputs/eth3d_training/vggt/raw results/eth3d_training/vggt/raw \
    outputs/eth3d_training/vggt/raw_canonical \
    results/eth3d_training/vggt/raw_canonical \
  --model dust3r \
    outputs/eth3d_training/dust3r/raw results/eth3d_training/dust3r/raw \
    outputs/eth3d_training/dust3r/raw_canonical \
    results/eth3d_training/dust3r/raw_canonical \
  --repair-prepared-root data/eth3d_training/raw_canonical \
  --png results/paper/qualitative_eth3d.png \
  --report results/paper/qualitative_eth3d.json

MPLBACKEND=Agg PYTHONPATH=src python3 scripts/render_qualitative_grid.py \
  dtu-held-out results/paper/qualitative_dtu.pdf \
  --model vggt \
    outputs/dtu/vggt/rectified_mechanism \
    results/dtu/vggt/rectified_mechanism \
    outputs/dtu/vggt/rectified_canonical \
    results/dtu/vggt/rectified_canonical \
  --model dust3r \
    outputs/dtu/dust3r/rectified_mechanism \
    results/dtu/dust3r/rectified_mechanism \
    outputs/dtu/dust3r/rectified_canonical \
    results/dtu/dust3r/rectified_canonical \
  --repair-prepared-root data/dtu/rectified_canonical \
  --png results/paper/qualitative_dtu.png \
  --report results/paper/qualitative_dtu.json
```

Each PDF contains all 4 scenes x 2 models x 3 registered variants.  The JSON
report hashes every prediction, metadata file, evaluation, validity mask, and
rendered artifact.  Copy the PDFs, previews, and reports back to `vircs` with a
binary-safe transfer and compare their SHA-256 values before committing them.

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

After the complete ETH3D and DTU summaries exist for both models, compute all
severity, matched-scope, and cross-dataset gates from the scene rows in one
command:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_mechanism_study.py \
  configs/eth3d_mechanism_variants.json \
  results/mechanism/analysis.json \
  --summary vggt eth3d results/eth3d_training/vggt/raw_mechanism/summary.json \
  --summary dust3r eth3d results/eth3d_training/dust3r/raw_mechanism/summary.json \
  --summary vggt dtu results/dtu/vggt/rectified_mechanism/summary.json \
  --summary dust3r dtu results/dtu/dust3r/rectified_mechanism/summary.json \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17 --rotation-threshold 2.0 --depth-threshold 0.05
```

The report recomputes paired identity deltas from every scene row rather than
trusting pre-aggregated values. A dataset supports a family only when every
supplied model on that dataset crosses a registered point-estimate threshold;
the full hypothesis requires two such families on two datasets. Confidence
intervals remain visible and may narrow the prose even when a point-estimate
gate passes.

Render the registered retained-fraction curves from that machine-readable
analysis (the same command also works on an ETH3D-only development report):

```bash
PYTHONPATH=src python3 scripts/plot_mechanism_severity.py \
  results/mechanism/analysis.json paper/figures/mechanism_severity.pdf \
  --png results/mechanism/mechanism_severity.png
```

Build the compact cross-dataset table source from the four confirmatory
variants. ETH3D point distances are converted from meters to millimeters;
focal, principal-point, and depth deltas are converted to percentage points.
The output retains every interval and the SHA-256 of each source summary:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_cross_dataset_table.py \
  results/mechanism/cross_dataset_table.json \
  --summary vggt eth3d results/eth3d_training/vggt/raw_mechanism/summary.json \
  --summary dust3r eth3d results/eth3d_training/dust3r/raw_mechanism/summary.json \
  --summary vggt dtu results/dtu/vggt/rectified_mechanism/summary.json \
  --summary dust3r dtu results/dtu/dust3r/rectified_mechanism/summary.json
```

Build the unified compute source without conflating model-only and end-to-end
timings.  First derive the complete legacy ETH3D metadata summaries; their
model compute and VRAM remain valid, while missing schema-1.2 end-to-end time
is explicitly labeled `legacy_unavailable`:

```bash
./scripts/run_eth3d_compute_summary.sh vggt
./scripts/run_eth3d_compute_summary.sh dust3r

PYTHONPATH=src .venv/bin/python scripts/summarize_compute_table.py \
  results/paper/compute_table.json \
  --sweep vggt eth3d-training-raw mechanism \
    results/eth3d_training/vggt/inference_compute.json \
  --sweep dust3r eth3d-training-raw mechanism \
    results/eth3d_training/dust3r/inference_compute.json \
  --sweep vggt dtu-held-out mechanism \
    results/dtu/vggt/inference_compute.json \
  --sweep dust3r dtu-held-out mechanism \
    results/dtu/dust3r/inference_compute.json \
  --sweep vggt dtu-held-out-repair canonical_repair \
    results/dtu/vggt/repair_inference_compute.json \
  --sweep dust3r dtu-held-out-repair canonical_repair \
    results/dtu/dust3r/repair_inference_compute.json \
  --canonicalization \
    results/dtu/rectified_canonical_preparation_compute.json \
  --repair-ablation artifacts/eth3d_repair_seed17/ablation_summary.json
```

The output keeps model load, model-only median/p90/total, schema-1.2
end-to-end median/p90/total, view count, and peak GiB separate.  The
canonicalization section separately reports decode, inverse warp, mask/PNG
write, and manifest-update wall time by source variant.  Never add that
preprocessing time into model compute or present legacy ETH3D model timing as
end-to-end timing.

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

After the neutral-gray predictions finish, compare the original and
canonical-identity prediction arrays before GT evaluation:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_prediction_repeat.py \
  outputs/eth3d_training/vggt/raw \
  outputs/eth3d_training/vggt/raw_canonical \
  --scenes courtyard delivery_area electro facade kicker meadow office \
    pipes playground relief relief_2 terrace terrains \
  --variant identity \
  --output results/repair/eth3d_vggt_identity_repeat_audit.json
```

Repeat for DUSt3R. The audit requires identical protocol metadata and records
exact equality plus maximum numerical drift for every NPZ array. Runtime and
path fields are intentionally excluded. A mismatch remains visible and does
not get replaced by an assumed zero clean cost.

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

Repeat without changing thresholds for DUSt3R.  DTU uses the strict dedicated
wrappers above and the same unchanged thresholds.  Aggregate gap
recovery is the median paired recovered gap divided by the median paired
corruption gap; bootstrap replicates resample scenes and recompute that ratio.
The output never clips recovery to `[0, 1]`: negative repair and better-than-
identity overshoot remain visible. A non-positive or noise-floor corruption
gap is marked undefined. The registered point-estimate gate is at least 30%
recovery and at most 2% median relative clean degradation; confidence-bound
versions are reported separately and never silently substituted for the
registered point-estimate rule. Pose-only records keep depth unavailable.

After the three fill reports and consensus report are frozen for each model,
build the single paper-ablation source. It validates model, dataset, scene,
candidate-order, analytic-baseline, and protocol agreement before exposing
rotation/depth recovery, compute, peak VRAM, and selection frequencies:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_repair_ablation.py \
  results/repair/ablation_summary.json \
  --model vggt results/repair/vggt_neutral.json \
    results/repair/vggt_black.json results/repair/vggt_mean.json \
    results/repair/vggt_consensus.json \
  --model dust3r results/repair/dust3r_neutral.json \
    results/repair/dust3r_black.json results/repair/dust3r_mean.json \
    results/repair/dust3r_consensus.json
```

The paper method overview is generated as vector artwork from source rather
than edited by hand:

```bash
MPLBACKEND=Agg python3 scripts/draw_method_overview.py \
  paper/figures/method_overview.pdf \
  --png results/paper/method_overview.png
```

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

The analyzer stores both score-ranked and oracle risk--coverage curves. Render
the frozen held-out panels without re-entering any values by hand:

```bash
PYTHONPATH=src python3 scripts/plot_risk_coverage.py \
  paper/figures/dtu_risk_coverage.pdf \
  --panel "VGGT — DTU" \
    results/reliability/dtu_vggt_seed17/rotation_disagreement.json \
    results/reliability/dtu_vggt_seed17/rotation_native_uncertainty.json \
    results/reliability/dtu_vggt_seed17/cases.json \
  --panel "DUSt3R — DTU" \
    results/reliability/dtu_dust3r_seed17/rotation_disagreement.json \
    results/reliability/dtu_dust3r_seed17/rotation_native_uncertainty.json \
    results/reliability/dtu_dust3r_seed17/cases.json \
  --png results/reliability/dtu_risk_coverage.png
```

Finally, audit claim promotion separately from evidence completeness.  This
command reopens the reliability cases and checks their SHA-256, exact 22 x 4
design, strict threshold, score fields, scene-cluster bootstrap, and model
identity.  It also requires the complete 13/22-scene, 11-variant mechanism
matrix and the unchanged DTU repair gates.  A failed AUROC, mechanism, or
repair gate remains a complete negative result and does not make the audit
fail:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_final_claims.py \
  results/paper/final_claim_audit.json \
  --mechanism results/mechanism/analysis.json \
  --reliability vggt \
    results/reliability/dtu_vggt_seed17/rotation_disagreement.json \
    results/reliability/dtu_vggt_seed17/rotation_native_uncertainty.json \
  --reliability dust3r \
    results/reliability/dtu_dust3r_seed17/rotation_disagreement.json \
    results/reliability/dtu_dust3r_seed17/rotation_native_uncertainty.json \
  --repair vggt results/repair/dtu_vggt_neutral_gray.json \
  --repair dust3r results/repair/dtu_dust3r_neutral_gray.json \
  --expected-models vggt dust3r \
  --detector-auroc-threshold 0.75 \
  --repair-recovery-threshold 0.30 --clean-relative-threshold 0.02
```

Use `evidence_complete` to decide whether final writing may begin; use the four
separate `claim_gates` only to decide which claims are promoted.  The audit
deliberately does not turn those gates into an automated reviewer score.

Freeze the final evidence only after the claim audit and all four figures are
present.  The bundle manifest copies the 572 lightweight per-scene GT records,
summaries, reliability cases, audits, compute, and figure assets.  It hashes
all 572 large prediction NPZ files into `BUNDLE.json` without copying them into
Git.  Exact counts, safe relative targets, copied bytes, and bundle checksums
are mandatory:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_evidence_bundle.py \
  configs/dtu_evidence_bundle.json results/frozen_dtu_seed17 --resume
```

An interrupted atomic copy can resume only when its temporary bytes match the
source. Existing target drift or an unexpected extra file aborts the freeze.
Transfer `results/frozen_dtu_seed17` to `vircs` byte-for-byte, verify its
`SHA256SUMS`, and commit it as `artifacts/dtu_seed17`; never copy the large NPZ
files themselves into Git.
