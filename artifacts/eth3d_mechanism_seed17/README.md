# ETH3D mechanism sweep, seed 17

This directory freezes the complete 13-scene, 11-variant ETH3D raw
ground-truth sweep for VGGT and DUSt3R. Each model has 143 paired evaluations;
the two models remain separate. The design and thresholds were frozen in
`configs/eth3d_mechanism_variants.json` before these additional severity and
crop-scope outcomes were inspected.

## Main result

- Independently shifted asymmetric crops degrade rotation monotonically as
  retained fraction falls from 90% to 75% to 60%. The paired rotation deltas
  are 1.31, 4.61, and 10.46 degrees for VGGT and 1.19, 4.52, and 10.89 degrees
  for DUSt3R.
- A shared off-center crop window is a second family crossing the registered
  2-degree point-estimate threshold on both models at 75% and 60% retention.
  At 75%, its paired rotation delta is 2.38 degrees (95% CI [1.12, 2.77]) for
  VGGT and 2.18 degrees ([0.88, 2.38]) for DUSt3R. At 60%, the corresponding
  values are 3.71 degrees ([2.34, 6.29]) and 3.30 degrees ([1.50, 5.56]).
- View-dependent crop windows are nevertheless consistently worse than a
  shared window at matched retention. The independent-minus-shared rotation
  contrasts have intervals above zero at all three severities for both
  models; the point estimates at 90%, 75%, and 60% are 0.58, 2.01, and 4.37
  degrees for VGGT and 0.41, 2.60, and 7.07 degrees for DUSt3R.
- Center crops and letterbox do not cross either registered rotation/depth
  threshold on ETH3D. DUSt3R center-crop rotation is not monotone, which is
  retained as a negative mechanism result.

This establishes two transform families on one dataset, not the registered
two-family, two-dataset hypothesis. DTU remains required. Threshold crossing
uses the frozen point-estimate rule; confidence intervals remain reported and
are not substituted for that rule.

All camera and depth scene metrics are complete. The 60% independent-crop
point-map alignment is undefined for one VGGT scene (`facade`) and one DUSt3R
scene (`office`); those values are not imputed, and their point-map bootstrap
metrics remain explicitly unavailable.

## Files and integrity

- `vggt_summary.json`: complete evaluator output; SHA-256
  `3090245a5d1e41a42d544f1bc8481ddb98ea1a110f91f7d23ccfbed2880cd670`.
- `dust3r_summary.json`: complete evaluator output; SHA-256
  `86422b149fc064631dcee998f1b87be344c64db8fb7b9ac5c8b751af8129fd19`.
- `analysis.json`: paired severity, matched-scope, and conservative
  cross-dataset gates; SHA-256
  `5e2c26826b4d535dbd3db62b6c77697bb6926c48253516b841954d39e10ded24`.
- Frozen variant config SHA-256:
  `2219be64a9c0e829ce619fe8aa431f53aad8c7ef7695a5bd570d48054187100e`.

The analysis uses a 10,000-replicate percentile scene bootstrap, 95%
confidence, and seed 17. Source-summary and config hashes are embedded in
`analysis.json`.
