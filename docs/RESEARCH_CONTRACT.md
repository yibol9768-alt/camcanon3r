# CamCanon3R research contract

Updated: 2026-08-03  
Target: 3DV 2027, paper deadline 2026-08-28 11:00 PDT  
Status: frozen pre-result kill-test contract

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
- asymmetric crop followed by resize;
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
| Common preprocessing breaks 3D equivariance | paired multi-model, multi-dataset geometry results | positive VGGT room pilot; confirmation needed |
| Native confidence misses failures | calibration and risk-coverage comparison | needs evidence |
| Disagreement detects failures | held-out AUROC with confidence intervals | needs evidence |
| CamCanon repairs geometry | paired baseline/ablation results and compute cost | needs evidence |

No abstract or introduction may promote a claim whose status remains "needs
evidence."
