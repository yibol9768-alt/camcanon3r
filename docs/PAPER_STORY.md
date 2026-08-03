# Paper story and claim-evidence gates

## One-sentence story

Known image preprocessing changes the camera matrix, yet feed-forward 3D
systems can return different scene geometry after the exact camera update is
removed; CamCanon3R audits this equivariance contract and will promote
disagreement to a detector or repair signal only if ground-truth experiments
support that promotion.

## Introduction mini-outline and paragraph roles

1. **Opening/task:** preprocessing is a camera operation, so reconstruction
   should be invariant after exact camera correction and global gauge.
2. **Challenge:** user and model preprocessing compose, and naive comparisons
   confound model drift with incorrect coordinate bookkeeping.
3. **Method:** log and compose exact affines, map predictions to a common
   domain, and compare cameras/depth with gauge-invariant metrics.
4. **Evidence:** three-scene asymmetric-crop signal plus exact identity control;
   center-crop scene dependence narrows the mechanism hypothesis.
5. **Contributions/boundary:** audit formulation and protocol are supported;
   GT degradation, generality, detection, and repair are promotion gates.

## Reverse outline

| Paragraph | Topic sentence message | Evidence or explanation |
|---|---|---|
| 1 | Practical preprocessing changes the camera. | $P'=AK[R\mid t]$ and gauge requirement. |
| 2 | Auditing is non-trivial because transforms compose. | Hidden model maps, aspect ratio, common coordinates. |
| 3 | CamCanon3R makes the contract measurable. | Stored $A$, $B$, $BA$; relative pose and depth alignment. |
| 4 | Current evidence isolates view-dependent crop drift. | Three scenes and identity-repeat control. |
| 5 | Contributions are intentionally evidence bounded. | Supported protocol; explicit confirmatory gates. |

## Claim-evidence map

| Claim | Evidence | Status | Promotion gate |
|---|---|---|---|
| Exact preprocessing transforms define a 3D equivariance contract. | Camera algebra and unit-tested affine composition. | supported | none |
| VGGT is non-equivariant to view-dependent 75% crops in the tested setting. | Three scenes, all above 2° median relative rotation; exact repeat control. | supported for diagnostic setting | severity sweep for broader claim |
| Preprocessing reduces ground-truth reconstruction accuracy. | None yet. Cross-run disagreement is insufficient. | needs evidence | ETH3D pose/depth |
| The failure generalizes beyond VGGT. | None yet. | needs evidence | DUSt3R confirmatory matrix |
| Cross-transform disagreement detects high-error cases. | None yet. | needs evidence | held-out AUROC and risk--coverage |
| Consensus repairs geometry. | None yet. | needs evidence | paired gap recovery, clean cost, runtime |

## Self-review

- **Clarity:** each introduction paragraph has one explicit role.
- **Flow:** camera change -> audit ambiguity -> protocol -> evidence -> boundary.
- **Terminology:** use *preprocessing non-equivariance* for current results;
  reserve *accuracy degradation* for ground-truth comparisons.
- **Unsupported claims:** no current text claims multi-model generality, AUROC,
  or repair gains.
- **Missing evidence:** ETH3D GT, severity response, DUSt3R, reliability, repair,
  statistics, compute-normalized baselines, and qualitative geometry.

The working title remains audit-only.  Add "and Repairing" only after the
registered repair thresholds are met.
