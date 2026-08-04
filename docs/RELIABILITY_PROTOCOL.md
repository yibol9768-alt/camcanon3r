# Reliability and failure-detection protocol

Status: score construction and evaluator frozen; empirical detector claim
pending. ETH3D is development-only because its outcomes were inspected before
this score was frozen. DTU is the held-out promotion dataset.

## Case and split unit

One case is one matched `(model, dataset, scene, view set, transform)` result.
The ground-truth error and every candidate uncertainty score must refer to the
same case. Train, validation, threshold selection, and reporting splits are
made by scene; transformed versions of one scene never cross splits.
Models are evaluated separately: cases and scores from VGGT and DUSt3R are not
pooled.

Reliability is reported separately for relative rotation, translation
direction, depth AbsRel, and point-cloud metrics. Raw-error AURC values are
scale dependent and must not be pooled across metrics.

## Scores and failures

- For each candidate transform, cross-transform disagreement is the median
  disagreement between that candidate and every other registered transform of
  the same model, scene, and view set. Identity is one candidate and is not
  assigned a privileged zero score. Ground truth never enters score
  construction. Pairwise depth disagreement is symmetrized as the mean of the
  two directed, scale-aligned AbsRel values before the candidate median.
- Native uncertainty is the negative median `world_points_conf`, falling back
  to `depth_conf` only when the former is absent. The selected field and
  conversion are recorded in every case.
- The failure threshold is frozen before held-out evaluation. The evaluator
  uses the strict rule `error > threshold`, matching the research contract's
  “more than” language.
- AUROC is undefined rather than fabricated when a split contains only one
  failure class.

## Metrics

The primary held-out endpoint is relative-rotation failure above 2 degrees.
Depth AbsRel above 0.05 is secondary where registered depth GT exists.
Translation-direction and point-cloud reliability remain diagnostic until
their failure thresholds are frozen without inspecting held-out outcomes.

AUROC uses the Mann--Whitney rank definition with average ranks for tied
scores. Risk--coverage retains cases from lowest to highest uncertainty and
adds an entire equal-score block at once, so arbitrary row order cannot change
the curve. We report raw AURC, oracle AURC obtained by ranking on ground-truth
error, and excess AURC (`AURC - oracle AURC`). Lower AURC and excess AURC are
better; higher AUROC is better.

Intervals use a deterministic percentile cluster bootstrap over scenes. Each
replicate samples scenes with replacement and carries all cases from each
sampled scene. Replicates containing only one failure class are omitted from
the AUROC interval and counted explicitly. Fewer than ten scenes and more than
10% undefined AUROC replicates trigger machine-readable warnings.

## Input and command

Build complete cases from a prediction/evaluation pair before running the
evaluator. This ETH3D command is for pipeline validation and exploratory
analysis only:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_reliability_cases.py \
  outputs/eth3d_training/vggt/raw \
  results/eth3d_training/vggt/raw \
  results/reliability/eth3d_vggt_raw_seed17/cases.json \
  --variants identity center_crop_075 asymmetric_crop_075 letterbox_square \
  --model vggt --dataset eth3d-training-raw
```

The evaluator accepts JSON, JSONL, or CSV cases and dot-separated nested field
paths. The minimal flat schema remains:

```json
{"scene": "office", "error": 3.2, "uncertainty": 0.74}
```

Run one score and error metric at a time:

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

The detector claim is promoted only if held-out AUROC reaches 0.75 with its
interval, native-confidence and oracle comparisons, risk--coverage, and case
provenance all reported. No scoring, threshold, or split change is allowed
after inspecting DTU reliability outcomes. Implementing this evaluator and
reporting exploratory ETH3D values are not evidence that the gate passes.
