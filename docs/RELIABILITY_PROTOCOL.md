# Reliability and failure-detection protocol

Status: evaluator implemented; empirical detector claim pending.

## Case and split unit

One case is one matched `(model, dataset, scene, view set, transform)` result.
The ground-truth error and every candidate uncertainty score must refer to the
same case. Train, validation, threshold selection, and reporting splits are
made by scene; transformed versions of one scene never cross splits.

Reliability is reported separately for relative rotation, translation
direction, depth AbsRel, and point-cloud metrics. Raw-error AURC values are
scale dependent and must not be pooled across metrics.

## Scores and failures

- Cross-transform disagreement is oriented so larger means more uncertain.
- Native confidence is converted to uncertainty before comparison, with the
  conversion recorded in the case file.
- The failure threshold is frozen before held-out evaluation. The evaluator
  uses the strict rule `error > threshold`, matching the research contract's
  “more than” language.
- AUROC is undefined rather than fabricated when a split contains only one
  failure class.

## Metrics

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

The evaluator accepts JSON, JSONL, or CSV cases. The default fields are:

```json
{"scene": "office", "error": 3.2, "uncertainty": 0.74}
```

Run one score and error metric at a time:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_reliability.py \
  results/reliability/rotation_cases.jsonl \
  --failure-threshold 2.0 \
  --bootstrap-replicates 10000 --confidence-level 0.95 \
  --bootstrap-seed 17 \
  --output results/reliability/rotation_disagreement.json
```

The detector claim is promoted only if held-out AUROC reaches 0.75 with its
interval, native-confidence and oracle comparisons, risk--coverage, and case
provenance all reported. Implementing this evaluator is not evidence that the
gate passes.
