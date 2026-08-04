# CamCanon3R research contract

Updated: 2026-08-04
Target: 3DV 2027, paper deadline 2026-08-28 11:00 PDT  
Status: frozen contract; matched VGGT/DUSt3R ETH3D result complete, promotion gates active

## Paper claim

Feed-forward 3D reconstruction models should be equivariant to known,
invertible image preprocessing once the induced camera update is accounted for.
CamCanon3R measures violations of this contract and uses cross-transform
agreement to detect and repair unreliable predictions without ground-truth 3D
at test time.

The work does **not** claim generic uncertainty estimation, robustness to
unrelated views, or a new reconstruction backbone. The contribution boundary
is preprocessing-induced camera drift and its known geometric correction.

## Formal contract

Let a view have camera matrix `P = K [R | t]`. An invertible image-space affine
map `A` changes the image formation equation to `P' = A K [R | t]`. A model run
on transformed images should therefore recover the same extrinsics and scene
geometry, with intrinsics mapped by `K' = A K`, up to the model's unavoidable
global gauge. After mapping predictions back and applying one declared global
Sim(3) alignment, residual differences measure preprocessing non-equivariance.

## Falsifiable hypotheses

1. At least two common preprocessing families cause more than 2 degrees of
   median relative-pose degradation or more than 5% absolute depth AbsRel
   increase on at least two datasets.
2. The model's native confidence does not fully predict these failures.
3. Cross-transform disagreement predicts high-error cases with AUROC at least
   0.75 without ground-truth 3D at test time.
4. A training-free consensus repair recovers at least 30% of the original-to-
   transformed performance gap while keeping clean degradation below 2%.

If hypothesis 1 fails, the paper direction stops. If 1 holds but 3 or 4 fails,
the output is an evaluation paper only if the benchmark reveals a stable,
previously undocumented failure with clear practical consequences.

## Transform families

- center crop followed by resize;
- asymmetric crop followed by resize, with either one normalized off-center
  window shared across views or independently sampled windows per view;
- isotropic and anisotropic resize;
- aspect-preserving resize plus symmetric letterbox padding;
- different transforms per view in the same input set.

Every sample stores `A`, source/target resolution, transformed intrinsics,
inverse pixel map, seed, and interpolation mode. JPEG recompression, blur, and
photometric shifts are excluded from the core study because they confound the
camera-model intervention with appearance corruption.

## Models and data

The 72-hour pilot uses official VGGT weights and official example scenes. The
confirmatory minimum is VGGT plus DUSt3R on DTU and ETH3D. MASt3R is added only
after its license, weights, and evaluation path are verified. All model-specific
preprocessing is logged; hidden preprocessing is treated as part of the system.

Before inspecting any DTU ground-truth coordinates or model outcomes, the DTU
protocol is frozen to the standard 22 MVSNet evaluation scans and the canonical
pixelNeRF three-view indices 22, 25, and 28 (zero-based). These map to camera
IDs 23, 26, and 29 in the official Rectified archive, using lighting index 3.
DTU is the held-out detector-promotion dataset: models remain separate, the
primary failure is relative-rotation error strictly above 2 degrees, and no
score or threshold may change after its ground-truth outcomes are inspected.
The official archive byte lengths and ETags are frozen in
`configs/dtu_sources.json`; the full protocol is in
`configs/dtu_protocol.json`.

## Baselines

- official model preprocessing and inference;
- naive transformed input without camera correction;
- analytic `A @ K` correction only;
- canonical letterbox preprocessing;
- transform ensemble with native-confidence selection;
- oracle transform selection using ground-truth error (upper bound).

The proposed minimum is consensus selection/fusion after inverse camera and
pixel mapping. A learned adapter is allowed only if the training-free method
passes the kill-test and leaves a measurable residual gap.

## Metrics and statistics

Report focal relative error, normalized principal-point error, relative
rotation and translation-direction error, depth AbsRel, and aligned point-cloud
accuracy/completeness. Reliability uses risk-coverage curves and failure AUROC.
All comparisons are paired by scene/view set/transform and use scene-level
bootstrap confidence intervals. Raw scores are always reported next to relative
gap recovery.

For the ETH3D raw protocol, point maps are evaluated against a target cloud
backprojected from the selected raw z-depth maps with the official
`THIN_PRISM_FISHEYE` camera model. Predicted geometry is restricted to tensor
pixels that map to finite scan-supported raw pixels. One orientation-preserving
Sim(3), fitted only from predicted and ground-truth camera poses, maps the
prediction into metric coordinates: the chordal mean of paired camera
rotations fixes global rotation, then camera centers fit positive scale and
translation. Surface ground truth never optimizes this alignment. Both clouds
are voxelized at 1 cm, deterministically capped at
100,000 points, and evaluated with untruncated bidirectional nearest-neighbor
distance (prediction-to-GT accuracy and GT-to-prediction completeness). To
bound raw-resolution backprojection, each view is first deterministically
sampled to at most 100,000 finite supported pixels before pooling. This is a
raw-depth-derived point-map metric, not the official ETH3D MVS leaderboard
protocol.

All ETH3D camera matrices and image sizes are resolved per view. A scene is not
assumed to use one COLMAP camera ID, and mixed-camera four-view subsets remain
in the frozen design rather than being removed after inspection.

## 72-hour kill-test

1. **0-8 h:** install VGGT on my5090, cache weights, freeze three example scenes,
   and verify exact camera-transform unit tests.
2. **8-24 h:** run original plus center/asymmetric crop and letterbox variants;
   quantify camera and point-map disagreement.
3. **24-44 h:** implement inverse mapping, Sim(3) alignment, and native-confidence
   baselines; decide whether hypothesis 1 holds.
4. **44-60 h:** implement cross-transform disagreement and consensus selection;
   compute detection AUROC and recovered gap.
5. **60-72 h:** repeat three seeds, profile 32 GB VRAM/runtime, freeze the result
   table, and continue or terminate according to the thresholds above.

## Reviewer risks

1. **"This is just augmentation."** The intervention is a known change of the
   camera matrix, and the target is geometric equivariance rather than average
   corruption robustness.
2. **"The fix is only ensembling."** Report accuracy per unit of compute and
   compare selection, averaging, analytic correction, and an oracle upper bound.
3. **"Only synthetic transforms."** Validate against real image pipelines that
   independently crop or letterbox views, while keeping the controlled protocol
   as the causal test.
4. **"One backbone."** The confirmatory claim requires at least two model
   families and two geometry datasets.

## Claim-evidence map

| Claim | Required evidence | Current status |
|---|---|---|
| Common preprocessing breaks 3D equivariance | paired multi-model, multi-dataset geometry results | **Multi-model ETH3D raw GT supported for view-dependent 75% crop:** VGGT and DUSt3R each have 13 scenes and 52 complete evaluations with pose/intrinsics/depth/point-map metrics; a second dataset and a second transform family crossing the registered threshold remain required for the full hypothesis |
| Native confidence misses failures | calibration and risk-coverage comparison | needs evidence |
| Disagreement detects failures | held-out AUROC with confidence intervals | needs evidence |
| CamCanon repairs geometry | paired baseline/ablation results and compute cost | needs evidence |

## Frozen confirmatory snapshot: VGGT and DUSt3R on ETH3D raw

The first benchmark-scale result is frozen in
`artifacts/eth3d_vggt_raw_seed17/summary.json`.  It evaluates four fixed views
from each of all 13 ETH3D training scenes under identity, shared center crop,
view-dependent asymmetric crop, and square letterbox preprocessing.  The 52
runs are complete, every registered GT metric is defined, and uncertainty is a
10,000-replicate scene bootstrap with seed 17.

Relative to the paired identity run, the view-dependent 75% crop increases the
median-of-scene-median rotation error by 4.61 degrees (95% CI [3.13, 6.17]),
translation-direction error by 9.49 degrees ([5.06, 13.55]), depth AbsRel by
0.0230 ([0.0119, 0.0295]), point accuracy error by 0.259 m
([0.133, 0.589]), and point completeness error by 0.369 m
([0.161, 0.756]).  These intervals exclude zero.  The principal-point error
increase is 0.0371 ([0.0368, 0.0372]), directly connecting the intervention to
the registered camera mechanism.

The controls narrow the claim.  Letterbox point accuracy and completeness
deltas are -0.0074 m ([-0.0204, 0.0210]) and -0.0019 m
([-0.0288, 0.0228]), respectively, and its depth delta also includes zero.
Shared center crop has no detectable depth increase, but it increases point
completeness error by 0.145 m ([0.0979, 0.1772]) and rotation error by 0.480
degrees ([0.130, 0.787]).  Therefore the VGGT artifact supports a strong
view-dependent-crop failure, not a claim that every crop or resize is harmful;
the matched-model boundary is evaluated separately below.

The matched DUSt3R artifact is frozen in
`artifacts/eth3d_dust3r_raw_seed17/summary.json` under the same 13-scene,
four-variant, seed-17 evaluator.  The asymmetric crop increases rotation error
by 4.52 degrees ([2.13, 5.72]), translation-direction error by 10.99 degrees
([7.66, 12.37]), depth AbsRel by 0.0164 ([0.00275, 0.0273]), and point
completeness error by 0.416 m ([0.114, 0.610]).  Its point-accuracy delta is
0.252 m ([-0.015, 0.469]) and is therefore not promoted as decisive for this
model.  Shared center cropping decisively increases DUSt3R depth AbsRel by
0.00868 ([0.00286, 0.0133]) and completeness error by 0.202 m
([0.123, 0.362]), but it does not cross the registered two-degree pose or 5%
absolute-depth thresholds.  Letterbox depth and both point-geometry intervals
include zero.

Read together, the two artifacts establish a multi-model ETH3D accuracy
failure for the view-dependent crop, with consistent camera, depth, and
completeness effects.  They do not satisfy the full two-transform,
two-dataset hypothesis and do not support a claim that every metric degrades
for every model.

The aggregation code now enforces one identity per scene, a complete paired
scene/variant design, separation of pose-only and depth protocols, and
deterministic scene-level bootstrap intervals. This is statistical
infrastructure, not evidence that any pending hypothesis passes.

Reliability score construction and evaluation are now frozen. Each candidate's
score is its median disagreement with every other registered transform of the
same model and scene; identity is not assigned zero, pairwise depth is
symmetrized, and native uncertainty is negative median model confidence.
ETH3D is development-only because its outcomes were already inspected, while
DTU is the held-out promotion dataset. Evaluation uses average-rank AUROC for
tied scores, complete tie blocks for risk--coverage, oracle and excess AURC,
and a scene-cluster bootstrap. Single-class AUROC and invalid bootstrap
replicates remain explicit undefined values. Detector promotion still requires
held-out AUROC at least 0.75; implementation and exploratory ETH3D results alone
do not satisfy that gate.

The repair protocol is likewise frozen before any repaired prediction was
evaluated against ground truth. It uses only the cropped pixels and known
affines, compares neutral-gray, black, and image-mean canonical fills, and
selects the candidate with minimum median cross-fill rotation disagreement.
Native confidence and GT-oracle selection use the identical three predictions.
The exact tie break, compute accounting, 30% recovery gate, 2% clean-cost gate,
and requirement to beat the one-pass neutral-gray baseline are specified in
`docs/REPAIR_PROTOCOL.md` and `configs/repair_consensus_protocol.json`.

No abstract or introduction may promote a claim whose status remains "needs
evidence."
