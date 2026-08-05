# Fancy-method protocol: canonical orbit projection

Status: frozen before any projected prediction or projected ground-truth result.
The historical CamCanon3R mechanism, repair, and support-control outcomes were
already known. This chronology is a development study, not a prospective claim
about the earlier experiments.

## Objective

The frozen audit baseline at commit
`e03cbb34cad2a87ed9f12788b93dba3901ee9cd5` diagnoses a preprocessing-induced
camera failure and partially repairs orientation with a known affine inverse.
The new method must produce a genuinely new camera reconstruction. It may not
merely choose among predictions or rename the analytic repair.

Canonical orbit projection runs a frozen black-box reconstruction model over a
symmetric, support-preserving set of camera-canvas placements. It transports
the resulting camera graphs into a common quotient representation, robustly
projects them onto a single cycle-consistent camera graph, and returns new
world-to-camera extrinsics. Ground truth, uncropped source pixels, and updated
backbone weights are unavailable to the method.

## Input orbit and information boundary

The input is the registered neutral-gray canonical repair of
`asymmetric_crop_075`. The visible image and its validity mask are embedded on
a canvas 1.25 times larger at nine frozen placements. A member never resamples
the repaired image. Every RGB byte and mask byte appears exactly once, every
member has the same output size and fill count, and all views use the same
placement within one model run.

The orbit contains the center, four axial placements, and four corners. Every
noncentral placement has a frozen inverse partner. These pairs expose odd
camera-canvas response while keeping visible support and model weights fixed.
The full design is stored in `configs/orbit_projection_protocol.json`.

## Quotient camera representation

Let run `m` predict world-to-camera rotations `R_i^m` for views `i`. Its global
world frame is arbitrary, so absolute matrices from different runs cannot be
averaged directly. The observable edge rotation is

`Delta_ij^m = R_j^m (R_i^m)^T`.

It is invariant to a global change of world gauge. Canonical orbit projection
first computes an inverse-placement-pair chordal mean for every camera edge,
then applies Tukey biweighting to suppress runs whose complete edge graph is
inconsistent with the orbit center. Finally, weighted SO(3) synchronization
recovers one cycle-consistent set of absolute rotations with the first camera
fixed as gauge.

Camera centers are expressed in first-camera coordinates, translated so the
first center is zero, and normalized by the median nonzero pairwise baseline
within each run. A weighted geometric median produces canonical centers. The
projected translations follow analytically from `t_i = -R_i C_i`.

The initial promoted claim is camera orientation only. Depth and point maps are
not silently paired with projected cameras and are not generic repair outputs.

## Baselines

Every model and dataset is reported separately against:

1. the one-pass neutral-gray analytic repair;
2. selection of the orbit medoid;
3. an unweighted chordal projection;
4. native-confidence selection;
5. a ground-truth selector reported only as an upper bound.

The projected method receives exactly the same orbit predictions as all
matched-compute selection and averaging baselines.

## Falsification and promotion

Synthetic tests must first establish gauge invariance, run-order invariance,
exact recovery for a consistent orbit, valid SO(3) output, and robustness to a
minority outlier.

The empirical primary endpoint is median pairwise relative-rotation error. A
promoted multi-run method must satisfy all conditions below for VGGT and
DUSt3R on ETH3D and DTU:

1. reduce at least 15 percent of the residual gap from the one-pass analytic
   repair toward the identity baseline;
2. never increase the model-dataset median by more than 0.1 degrees;
3. beat or tie both the orbit medoid and the uniform projection;
4. use no GT, uncropped pixels, or result-dependent retry;
5. report 10,000-replicate scene-cluster intervals and complete compute.

Existing support-control predictions are retrospective feasibility data only.
They cannot promote the method. The nine-member ETH3D orbit is development
data. The implementation and configuration hashes must be frozen before the
first DTU orbit evaluation.

## Conditional single-pass student

An affine-aware graph corrector is attempted only if the multi-run projection
passes. Its teacher is the frozen projected camera graph. The student receives
one backbone camera prediction, registered affine parameters, valid-support
fractions, and native confidence. It predicts per-view SE(3) corrections and a
failure score using a permutation-equivariant graph network.

The student must retain at least 80 percent of the teacher's rotation gain with
one backbone forward pass. Failure of this stage leaves a valid multi-run
method but forbids a single-pass claim.

## Scope boundary

This work does not rewrite the paper. Code, protocols, tests, raw logs, frozen
results, and a method handoff are produced first. Paper framing resumes only
after user review.
