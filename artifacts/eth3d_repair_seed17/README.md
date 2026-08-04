# ETH3D canonical-camera repair study, seed 17

This directory freezes the registered three-fill repair study on all 13 ETH3D
training scenes for VGGT and DUSt3R. The repair receives only the cropped
images and their known affines. Neutral gray, black, and image mean were frozen
before repair GT evaluation; cross-fill consensus, native confidence, and the
GT oracle all select from the same three predictions. Models are never pooled.

## Primary rotation result

The one-pass neutral-gray analytic repair meets the registered point-estimate
gate on both models with zero measured identity clean cost:

| Model | Identity | Corrupt | Repaired | Gap recovery (95% CI) |
|---|---:|---:|---:|---:|
| VGGT | 0.805° | 5.274° | 1.449° | 0.966 [0.729, 1.130] |
| DUSt3R | 1.276° | 5.667° | 2.934° | 0.558 [0.267, 1.060] |

The frozen point-estimate rule is recovery at least 0.30 and relative clean
degradation at most 0.02. VGGT also clears the confidence-bound recovery gate;
DUSt3R does not. Black fill gives DUSt3R recovery 0.554 [0.360, 1.053], showing
fill sensitivity, but it does not replace the predeclared neutral-gray
single-pass baseline.

Cross-fill consensus passes all point-estimate gates for VGGT and slightly
reduces its repaired rotation error from 1.449° to 1.416°. It fails the frozen
multi-model promotion condition: for DUSt3R, consensus equals the neutral-gray
median error (2.934°), does not beat the one-pass analytic baseline, and costs
three model runs. The consensus selector chooses neutral gray in 9/13 scenes
and image mean in 4/13 for both models. This negative result is retained; no
consensus-repair claim is promoted.

## Geometry boundary

Canonicalization repairs camera orientation but is not a general geometry
repair. Every registered fill worsens median depth AbsRel for both models. For
neutral gray, depth gap recovery is -1.38 [-4.45, -0.52] for VGGT and -12.61
[-93.04, -5.87] for DUSt3R. Neutral gray improves VGGT point accuracy and
completeness, but DUSt3R point accuracy worsens and its completeness interval
crosses zero. The DUSt3R image-mean point metrics are undefined in one scene;
the complete-design aggregator records 12/13 valid scenes and performs no
subset bootstrap or imputation.

Accordingly, the supported claim is narrow: known-affine canonicalization
recovers a large fraction of crop-induced rotation error at zero clean-control
cost on both tested models. It does not support repaired depth, all-metric
geometry, or a promoted consensus selector. DTU remains required for
cross-dataset repair evidence.

## Integrity and compute

- All 52 identity images and masks in each preparation have full support; the
  neutral, black, and image-mean cropped repairs have exact support 0.5625.
- Identity-repeat audits compare 130 prediction arrays per model; all 130 are
  byte-for-byte equal for both VGGT and DUSt3R.
- Median model compute per scene is 0.155 s for one-pass VGGT and 0.490 s for
  three-fill selection; the corresponding DUSt3R values are 4.80 s and 15.05
  s. Model loading is reported separately in the JSON records.
- The frozen repair protocol SHA-256 is
  `8a737b193143951db4059fe34a6b5c1ba124a2d1caa0ed8309d9bf5788179c2c`.
- `ablation_summary.json` binds all eight source reports by SHA-256 and exposes
  the six fill/selector rows per model with rotation, depth, compute, VRAM, and
  selection-frequency fields for direct paper-table generation.
- `SHA256SUMS` covers every JSON artifact in this directory.

All intervals use 10,000 scene-bootstrap replicates, 95% confidence, and seed
17. Recovery is the median paired recovered gap divided by the median paired
corruption gap and is never clipped.
