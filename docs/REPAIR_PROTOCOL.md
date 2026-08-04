# Canonical repair and consensus protocol

Status: frozen before any repair ground-truth result was produced. Accuracy
results from the original ETH3D transformed runs were already known, so repair
evaluation on ETH3D is confirmatory but not held out. DTU remains held out.

## Inputs and fairness boundary

The repair receives only the registered `asymmetric_crop_075` images and their
known image-space affines. It never reads the corresponding untransformed
image pixels. The affine inverse maps visible pixels back to the original
camera canvas; pixels removed by cropping remain irrecoverable and are filled
deterministically. Every repaired image carries a binary validity mask and
visible-support fraction.

The clean control applies the identical canonicalization path to identity
inputs. The primary analytic baseline uses neutral-gray fill. Black and
per-image RGB-mean fill are sensitivity variants, not extra information.

## Training-free selection

The frozen three-candidate order is `neutral_gray`, `black`, then `image_mean`.
Each candidate is reconstructed independently. For every candidate, the
consensus score is the median relative-rotation disagreement with the other
two candidates, using all camera pairs. The candidate with minimum score is
selected; exact ties use the frozen order. Ground truth, original untransformed
pixels, and native confidence never enter this score.

The matched-compute baselines select from the same three candidates:

- native confidence chooses the largest median `world_points_conf`, falling
  back to `depth_conf` only when necessary;
- the oracle chooses the smallest ground-truth median relative-rotation error
  and is reported only as an upper bound;
- neutral-gray is the one-run analytic baseline, so its lower compute remains
  explicit rather than being called matched.

## Metrics and gates

Every selected prediction is evaluated by the same dataset GT evaluator as
the original identity and corrupted prediction. Scene-level aggregation uses
the paired median corruption gap and median recovered gap:

`recovery = median(corrupt - selected) / median(corrupt - identity)`.

Scene bootstrap replicates recompute the ratio. Recovery is not clipped;
negative values and overshoot remain visible. The pre-registered primary
point-estimate gate is relative-rotation recovery of at least 30%, with median
relative clean-control degradation at most 2%. Confidence-bound gates are
reported separately. Models and datasets are never pooled. A consensus claim
also requires it to outperform the neutral-gray analytic baseline; otherwise
the three-run method is reported as a negative result.

Runtime, peak VRAM, run count, selected candidate frequencies, native and
oracle selections, raw errors, and all undefined metrics must accompany any
repair claim.
