# Paper story and claim-evidence gates

## One-sentence story

Known image preprocessing changes the camera matrix, yet feed-forward 3D
systems can return different scene geometry after the exact camera update is
removed; CamCanon3R audits this equivariance contract, demonstrates a
multi-scene VGGT accuracy failure under view-dependent cropping, and will
promote disagreement to a detector or repair signal only if held-out evidence
supports that promotion.

## Introduction mini-outline and paragraph roles

1. **Opening/task:** preprocessing is a camera operation, so reconstruction
   should be invariant after exact camera correction and global gauge.
2. **Challenge:** user and model preprocessing compose, and naive comparisons
   confound model drift with incorrect coordinate bookkeeping.
3. **Method:** log and compose exact affines, map predictions to a common
   domain, and compare cameras/depth with gauge-invariant metrics.
4. **Evidence:** 13-scene ETH3D GT pose, intrinsics, depth, and point geometry;
   three-scene DUSt3R diagnostics support replication without substituting for
   matched GT.
5. **Contributions/boundary:** audit formulation and VGGT/ETH3D degradation are
   supported; multi-model/dataset generality, detection, and repair remain
   promotion gates.

## Reverse outline

| Paragraph | Topic sentence message | Evidence or explanation |
|---|---|---|
| 1 | Practical preprocessing changes the camera. | $P'=AK[R\mid t]$ and gauge requirement. |
| 2 | Auditing is non-trivial because transforms compose. | Hidden model maps, aspect ratio, common coordinates. |
| 3 | CamCanon3R makes the contract measurable. | Stored $A$, $B$, $BA$; relative pose and depth alignment. |
| 4 | Current GT evidence isolates view-dependent crop degradation. | 13 ETH3D scenes; paired identity, center-crop, asymmetric-crop, and letterbox runs. |
| 5 | Contributions are intentionally evidence bounded. | Supported VGGT result; explicit cross-model, cross-dataset, detection, and repair gates. |

## Claim-evidence map

| Claim | Evidence | Status | Promotion gate |
|---|---|---|---|
| Exact preprocessing transforms define a 3D equivariance contract. | Camera algebra and unit-tested affine composition. | supported | none |
| VGGT is non-equivariant to view-dependent 75% crops in the tested setting. | Three-scene cross-run diagnostic plus exact repeat control. | supported for diagnostic setting | severity sweep for broader mechanism claim |
| Confirmatory uncertainty is auditable. | Deterministic 10,000-replicate scene bootstrap across all 13 ETH3D training scenes. | supported for the frozen VGGT/ETH3D matrix | matched replication for each promoted model/dataset |
| View-dependent 75% crops reduce VGGT ground-truth reconstruction accuracy on ETH3D raw. | Paired 13-scene GT pose/intrinsics/depth/point evaluation; all registered deltas available and decisive degradation intervals exclude zero. | supported | none for this narrow model/dataset/transform claim |
| The diagnostic failure generalizes beyond VGGT. | Frozen DUSt3R three-scene matrix; asymmetric crop exceeds 2° rotation in all scenes. | supported for the three-scene diagnostic | benchmark-scale multi-model GT accuracy |
| Cross-transform disagreement detects high-error cases. | None yet. | needs evidence | held-out AUROC and risk--coverage |
| Reliability metrics are auditable. | Tie-aware AUROC, tie-invariant risk--coverage, oracle/excess AURC, and scene-cluster bootstrap are unit tested. | implemented, no effectiveness claim | populate with held-out GT cases |
| Analytic canonical-camera repair restores source pixel coordinates. | Unit-tested inverse affine, identity control, masks, and manifests. | implemented, no effectiveness claim | paired GT gap recovery and clean cost |
| Consensus repairs geometry. | None yet. | needs evidence | paired gap recovery beyond analytic repair, clean cost, runtime |

## Self-review

- **Clarity:** each introduction paragraph has one explicit role.
- **Flow:** camera change -> audit ambiguity -> protocol -> evidence -> boundary.
- **Terminology:** use *preprocessing non-equivariance* for cross-run
  diagnostics and *accuracy degradation* only for the frozen GT comparison.
- **Unsupported claims:** no current text claims multi-model or multi-dataset
  GT generality, AUROC, or repair gains.
- **Missing evidence:** matched DUSt3R GT, a second geometry dataset, severity
  response, reliability, repair, compute-normalized baselines, and qualitative
  geometry.
- **Statistical boundary:** the three-scene diagnostics remain descriptive;
  the 13-scene VGGT/ETH3D intervals support only their registered paired
  model/dataset/transform claims.

## Method reverse outline

| Subsection | First-message role | Design and evidence boundary |
|---|---|---|
| Overview | Separate coordinate intervention from model response. | Logged maps, common-domain audit, analytic repair. |
| Equivariance contract | Compose user and hidden model maps exactly. | $C_i=B_iA_i$; unit-tested bookkeeping. |
| Common-domain comparison | Remove pixel and 3D gauge before measuring drift. | Common support, relative pose, one depth scale. |
| Canonical-camera repair | Undo coordinates without claiming to restore lost pixels. | Inverse warp, neutral fill, validity mask, identity control. |
| Disagreement | Promotion depends on held-out GT and matched compute. | No detector or consensus claim yet. |

The working title remains audit-only.  Add "and Repairing" only after the
registered repair thresholds are met.

The final prose and evidence layout are calibrated against the frozen
three-paper ICLR comparison in `docs/ICLR_WRITING_BENCHMARKS.md`. Those papers
are structural benchmarks, not a shared numerical leaderboard.
