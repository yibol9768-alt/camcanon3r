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
the resulting camera graphs into a common quotient representation, learns the
local camera response to known canvas coordinates without GT, evaluates that
response at zero bias, and synchronizes it into one cycle-consistent camera
graph. That camera graph then fixes the otherwise arbitrary Sim(3) gauge of
each dense reconstruction. Registered point maps and intrinsics are fused only
on common source support to produce one coherent full reconstruction. Ground
truth, uncropped source pixels, and updated backbone weights are unavailable to
the method.

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

It is invariant to a global change of world gauge. The main method maps each
edge rotation into the tangent space of its chordal reference and fits its
response to the known two-dimensional canvas placement. Constant, affine, and
quadratic response bases are compared by leave-one-orbit-member-out geodesic
prediction error. A more complex basis is selected only if it reduces the
median error by at least five percent. Tukey weighting then removes members
whose complete camera graph disagrees with the selected response. The response
intercept at centered placement is mapped back to SO(3), and complete-graph
rotation synchronization recovers one cycle-consistent set of absolute
rotations with the first camera fixed as gauge.

The centered member is a no-GT trust anchor. It retains a minimum regression
weight of four. If the fitted zero-bias camera graph moves more than two
degrees from that observed centered graph, the response is declared unsafe and
falls back to the inverse-pair robust group projection. The fallback and the
unclamped response remain recorded. This rule prevents a smooth response basis
from treating a direct center observation as an outlier.

An inverse-placement-pair robust mean remains a strong matched-compute baseline.
It is not the promoted method because group averaging and rotation averaging
are established components. The response field must beat or tie this baseline.

Camera centers are expressed in first-camera coordinates, translated so the
first center is zero, and normalized by the median nonzero pairwise baseline
within each run. A weighted geometric median produces canonical centers. The
projected translations follow analytically from `t_i = -R_i C_i`.

## Camera-constrained common-support geometry

The primary full output is not formed by silently pairing old point maps with
new cameras. For every orbit member, the method estimates a positive Sim(3)
from its predicted camera orientations and centers to the projected camera
graph. The same predicted transform maps that member's world point map into
the canonical world gauge. Logged source-to-model affines establish exact
cross-run pixel correspondences on the centered member's model grid.

Fusion is restricted by the canonical repair mask, so pixels removed by the
original crop remain undefined. Native confidence is normalized within each
member, multiplied by the response-field member weight, and used in a
geometric median over aligned 3D points. At least three finite members are
required. Intrinsics are first mapped back to source coordinates, fused by a
weighted median, and then carried to the reference model grid. Depth is
derived by transforming the fused point map through the projected cameras.
This produces a camera, intrinsic, depth, and point-map tuple with one recorded
provenance chain and no GT input.

Camera orientation remains the promotion endpoint for the response field.
Depth and point metrics are a separate full-reconstruction endpoint and are
reported against identity and one-pass analytic repair without imputing lost
support.

## Baselines

Every model and dataset is reported separately against:

1. the one-pass neutral-gray analytic repair;
2. selection of the orbit medoid;
3. inverse-pair robust group projection;
4. an unweighted chordal projection;
5. native-confidence selection;
6. a ground-truth selector reported only as an upper bound.

The projected method receives exactly the same orbit predictions as all
matched-compute selection and averaging baselines.

The full reconstruction additionally compares the camera-constrained response
fusion with identity and one-pass analytic repair for focal length, principal
point, depth where available, and point accuracy/completeness.

## Falsification and promotion

Synthetic tests must first establish gauge invariance, run-order invariance,
exact recovery for a consistent orbit, valid SO(3) output, robustness to a
minority outlier, exact camera-derived Sim(3) recovery, registered point-map
fusion, and preservation of undefined support.

The empirical primary endpoint is median pairwise relative-rotation error. A
promoted multi-run method must satisfy all conditions below for VGGT and
DUSt3R on ETH3D and DTU:

1. reduce at least 15 percent of the residual gap from the one-pass analytic
   repair toward the identity baseline;
2. never increase the model-dataset median by more than 0.1 degrees;
3. beat or tie the robust group projection, orbit medoid, and uniform
   projection;
4. use no GT, uncropped pixels, or result-dependent retry;
5. report 10,000-replicate scene-cluster intervals and complete compute.

Existing support-control predictions are retrospective feasibility data only.
They cannot promote the method. The nine-member ETH3D orbit is development
data. The implementation and configuration hashes must be frozen before the
first DTU orbit evaluation.

The full geometry endpoint has its own gate. Relative to one-pass analytic
repair, no reported intrinsic or dataset-available geometry metric may worsen
by more than two percent, and at least one depth or point metric must improve
by five percent. ETH3D uses depth AbsRel plus point accuracy and completeness;
DTU uses point accuracy and completeness. Passing this gate does not substitute
for the separate camera-response promotion gate.

## Conditional single-pass student

An affine-aware graph corrector is attempted only if the response-field
projection
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
