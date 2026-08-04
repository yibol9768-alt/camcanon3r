# Paper story and claim-evidence gates

## One-sentence story

Known image preprocessing changes the camera matrix, yet feed-forward 3D
systems can return different scene geometry after the exact camera update is
removed; CamCanon3R audits this equivariance contract, isolates matched
VGGT/DUSt3R off-center-crop failures and their view-dependent amplification on ETH3D, and will
promote disagreement to a detector or repair signal only if held-out evidence
supports that promotion.

## Introduction mini-outline and paragraph roles

1. **Opening/task:** preprocessing is a camera operation, so reconstruction
   should be invariant after exact camera correction and global gauge.
2. **Challenge:** user and model preprocessing compose, and naive comparisons
   confound model drift with incorrect coordinate bookkeeping.
3. **Method:** log and compose exact affines, map predictions to a common
   domain, and compare cameras/depth with gauge-invariant metrics.
4. **Evidence:** separate 13-scene VGGT and DUSt3R ETH3D GT pose, intrinsics,
   depth, and point geometry, plus severity/scope and exact-repeat controls.
5. **Contributions/boundary:** audit formulation and multi-model ETH3D
   degradation are supported; cross-dataset generality, detection, and repair
   remain promotion gates.

## Reverse outline

| Paragraph | Topic sentence message | Evidence or explanation |
|---|---|---|
| 1 | Practical preprocessing changes the camera. | $P'=AK[R\mid t]$ and gauge requirement. |
| 2 | Auditing is non-trivial because transforms compose. | Hidden model maps, aspect ratio, common coordinates. |
| 3 | CamCanon3R makes the contract measurable. | Stored $A$, $B$, $BA$; relative pose and depth alignment. |
| 4 | Current GT evidence isolates off-center crop degradation and view-dependent amplification. | 13 ETH3D scenes, 11 variants per model; separate paired VGGT and DUSt3R aggregates. |
| 5 | Contributions are intentionally evidence bounded. | Two supported ETH3D transform families; explicit cross-dataset, detection, and repair gates. |

## Claim-evidence map

| Claim | Evidence | Status | Promotion gate |
|---|---|---|---|
| Exact preprocessing transforms define a 3D equivariance contract. | Camera algebra and unit-tested affine composition. | supported | none |
| VGGT is non-equivariant to view-dependent 75% crops in the tested setting. | Three-scene cross-run diagnostic plus exact repeat control. | supported for diagnostic setting | severity sweep for broader mechanism claim |
| Confirmatory uncertainty is auditable. | Deterministic 10,000-replicate scene bootstrap across all 13 ETH3D training scenes, reported separately for VGGT and DUSt3R. | supported for both frozen ETH3D matrices | matched replication for each additional promoted dataset |
| View-dependent 75% crops reduce VGGT ground-truth reconstruction accuracy on ETH3D raw. | Paired 13-scene GT pose/intrinsics/depth/point evaluation; all registered deltas available and decisive degradation intervals exclude zero. | supported | none for this narrow model/dataset/transform claim |
| View-dependent 75% crops reduce DUSt3R ground-truth reconstruction accuracy on ETH3D raw. | Paired 13-scene GT evaluation; rotation, translation, depth, principal-point, and completeness deltas exclude zero, while point accuracy and focal deltas do not. | supported with metric-level boundary | none for this narrow model/dataset/transform claim |
| The GT failure generalizes beyond one model. | Separate complete VGGT and DUSt3R ETH3D aggregates under protocol 2.1. | supported on ETH3D | matched second-dataset replication |
| Shared off-center crops form a second failure family, while view-dependent offsets amplify it. | Frozen 90/75/60% severity and matched shared/independent-window sweep; all six independent-minus-shared rotation intervals exclude zero. | supported on ETH3D for both models | matched DTU replication for the full hypothesis |
| Cross-transform disagreement detects high-error cases. | None yet. | needs evidence | held-out AUROC and risk--coverage |
| Reliability metrics are auditable. | Tie-aware AUROC, tie-invariant risk--coverage, oracle/excess AURC, and scene-cluster bootstrap are unit tested. | implemented, no effectiveness claim | populate with held-out GT cases |
| Analytic canonical-camera repair restores source pixel coordinates. | Unit-tested inverse affine, identity control, masks, and manifests. | implemented, no effectiveness claim | paired GT gap recovery and clean cost |
| Consensus repairs geometry. | None yet. | needs evidence | paired gap recovery beyond analytic repair, clean cost, runtime |

## Self-review

- **Clarity:** each introduction paragraph has one explicit role.
- **Flow:** camera change -> audit ambiguity -> protocol -> evidence -> boundary.
- **Terminology:** use *preprocessing non-equivariance* for cross-run
  diagnostics and *accuracy degradation* only for the frozen GT comparison.
- **Unsupported claims:** no current text claims multi-dataset GT generality,
  that every transform/metric degrades, AUROC, or repair gains.
- **Missing evidence:** a second geometry dataset, held-out reliability,
  repair, compute-normalized baselines, and qualitative geometry.
- **Statistical boundary:** the three-scene diagnostics remain descriptive;
  the two separate 13-scene ETH3D aggregates support only their registered
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
