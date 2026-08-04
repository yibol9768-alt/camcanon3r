# ICLR writing and evidence benchmarks

Updated: 2026-08-04

These three papers are structural writing and evidence benchmarks for the
CamCanon3R draft. They are not interchangeable numerical baselines: each solves
a different task and uses a different training and evaluation protocol. We use
them to calibrate claim scope, experimental breadth, ablations, and limitations,
not to manufacture a direct leaderboard.

## Frozen comparison set

1. **π³: Permutation-Equivariant Visual Geometry Learning**, ICLR 2026.
   Primary sources: [OpenReview](https://openreview.net/forum?id=DTQIjngDta),
   [arXiv](https://arxiv.org/abs/2507.13347).
2. **Cameras as Rays: Pose Estimation via Ray Diffusion**, ICLR 2024 Oral.
   Primary sources: [OpenReview](https://openreview.net/forum?id=EanCFCwAjM),
   [arXiv](https://arxiv.org/abs/2402.14817).
3. **PF-LRM: Pose-Free Large Reconstruction Model for Joint Pose and Shape
   Prediction**, ICLR 2024 Spotlight.
   Primary sources: [OpenReview](https://openreview.net/forum?id=noe76eRcPC),
   [arXiv](https://arxiv.org/abs/2311.12024).

The set is frozen before the ETH3D confirmatory results. π³ is the closest
equivariance paper; Cameras as Rays is the closest camera-representation and
uncertainty paper; PF-LRM is the closest joint pose/reconstruction systems
paper with explicit robustness tests.

## Structural comparison

| Dimension | π³ | Cameras as Rays | PF-LRM | CamCanon3R requirement |
|---|---|---|---|---|
| Opening problem | Fixed reference views introduce order sensitivity. | Global pose parameters are a poor learned representation for sparse-view ambiguity. | Sparse unposed views break conventional pose-first reconstruction. | Resizing, cropping, and padding are camera operations, but pipelines treat them as plumbing. |
| Central mechanism | Reference-free, permutation-equivariant pose and point-map representation. | Patch-wise ray bundles, with regression and diffusion variants. | Joint image/triplane tokens plus differentiable PnP. | Register every user/model affine, undo known camera changes, and measure residual 3D drift. |
| Main evidence pattern | Pose, point maps, depth, order robustness, component ablations, speed. | Pose metrics across 2--8 views, seen/unseen categories, representation ablation, uncertainty examples. | Pose and rendering on five unseen datasets, variable-view and mask-noise robustness, objective/model-size ablations. | Two-model diagnostic first; then paired multi-scene ETH3D GT pose, intrinsics, depth/point geometry, detection, repair, and compute. |
| Robustness intervention | Input ordering/reference-frame selection. | Sparse-view ambiguity and number of input rays/views. | View count, segmentation noise, and lighting. | Known image-coordinate transforms with exact induced camera update. |
| Limitations practice | Names material/geometry failure modes and artifacts. | States that explicit geometric consistency is not enforced. | Names background, appearance, resolution, known-intrinsics, and pose-supervision assumptions. | State missing pixels, model/license scope, undefined metrics, negative repair, and failed promotion gates explicitly. |

## Scientific positioning to preserve

- **Against π³:** permutation equivariance concerns the order and reference
  frame assigned to a fixed image set. CamCanon3R concerns equivariance to a
  known per-image coordinate transform. These are complementary axes; do not
  imply that π³ already tested crop/resize camera equivariance.
- **Against Cameras as Rays:** ray diffusion represents pose uncertainty by
  sampling learned modes. CamCanon3R's cross-transform disagreement is a
  deterministic audit signal. Do not call it uncertainty estimation until
  held-out GT AUROC and risk--coverage pass the registered gate.
- **Against PF-LRM:** PF-LRM conditions on and explicitly lists known intrinsics
  as an assumption. CamCanon3R audits an upstream systems boundary where image
  preprocessing deterministically changes those intrinsics. This is a precise
  contrast, not a claim that PF-LRM is implemented incorrectly.

## Writing pattern for the final draft

Each major section must follow `claim -> mechanism -> evidence -> boundary`.

1. **Abstract:** one problem sentence, one exact contract, one protocol
   sentence, two or three decisive GT numbers, then the narrow conclusion. No
   pending result or future tense.
2. **Introduction:** identify one overlooked assumption; explain why ordinary
   corruption tests do not answer it; state the exact intervention; preview the
   strongest paired GT evidence; list only contributions that passed gates.
3. **Related work:** compare assumptions and intervention axes, not paper
   popularity. The three paragraphs should end by stating the unfilled gap.
4. **Experiments:** begin with frozen datasets/view selection, then baselines
   and metrics, main GT result, cross-model result, severity/mechanism ablation,
   reliability, repair, compute, and failures. Every relative claim must retain
   its raw metric.
5. **Conclusion and limitations:** restate only promoted claims. Failed AUROC or
   repair thresholds remain results, not text to hide.

## Evidence-gap checklist before reviewer red-team

- [x] At least 10 held-out ETH3D scenes with deterministic view selection and
  complete paired variants.
- [x] Absolute GT pose, intrinsics, raw-depth, and aligned point-cloud
  accuracy/completeness, plus deltas from identity.
- [x] VGGT and DUSt3R reported separately on a matched GT protocol.
- [ ] At least two transform families cross the frozen degradation threshold;
  otherwise narrow hypothesis 1 rather than changing the threshold.
- [ ] Severity response and view-dependent-versus-shared crop ablation test the
  principal-point mechanism.
- [ ] Native confidence, transform disagreement, AUROC, risk--coverage, and
  scene-cluster intervals on held-out cases.
- [ ] Analytic repair, consensus, clean-cost control, oracle selection, raw gap
  recovery, runtime, VRAM, and accuracy per unit compute.
- [ ] Representative success and failure cases chosen without outcome-driven
  scene filtering.
- [ ] Abstract, title, contributions, tables, and conclusion regenerated only
  from frozen result artifacts.
- [ ] Independent reviewer red-team finds no diagnostic/GT conflation, hidden
  selection, missing undefined values, or unsupported generalization.

## Target-score interpretation

The requested ICLR-style 6--8 target is treated as an honest reviewer estimate,
not an optimization label. A 6-level case requires a clear new systems finding,
correct evaluation, multi-model GT evidence, and useful negative boundaries. A
7--8 estimate additionally requires stable breadth, a convincing detector or
repair contribution, strong ablations, and a paper whose central claim survives
the most skeptical alternative explanations. The project will report the
estimate and residual objections after red-team; it will not declare the score
in advance.
