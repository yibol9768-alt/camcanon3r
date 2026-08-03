# Frozen multi-scene VGGT diagnostic

Date: 2026-08-03
Views per scene: 4
Seed: 17
Model: official VGGT-1B weights

This diagnostic measures disagreement between transformed and identity model
runs after exact affine bookkeeping.  It demonstrates preprocessing
**non-equivariance**, not degradation against ground-truth geometry.

| Scene | Transform | Median rotation | Median translation direction | Mean aligned depth AbsRel |
|---|---|---:|---:|---:|
| kitchen | asymmetric crop, 75% | 3.787° | 3.767° | 3.445% |
| llff_fern | asymmetric crop, 75% | 5.018° | 18.440° | 4.310% |
| room | asymmetric crop, 75% | 6.297° | 32.283° | 9.939% |
| kitchen | center crop, 75% | 0.154° | 0.598° | 3.809% |
| llff_fern | center crop, 75% | 0.239° | 0.725° | 3.314% |
| room | center crop, 75% | 3.876° | 18.002° | 8.666% |
| room | identity repeat | 0.000° | 0.0000004° | effectively 0% |

Across scenes, the asymmetric crop has a median-of-scene median rotation
disagreement of 5.018° and exceeds the frozen 2° threshold in all three
scenes.  The center crop exceeds it in one of three scenes.  Letterbox padding
exceeds it in none of the three scenes.

The narrow mechanism hypothesis is therefore that **view-specific
principal-point offsets** are a stable source of drift.  The evidence does not
support the broader statement that every crop or resize degrades geometry.
Severity sweeps, ETH3D ground truth, and a second reconstruction model remain
required.

Machine-readable evidence is stored in
[`artifacts/pilot_multiscene_seed17`](../artifacts/pilot_multiscene_seed17).
