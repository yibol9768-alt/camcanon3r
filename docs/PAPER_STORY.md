# Paper story and claim-evidence gates

Updated: 2026-08-04, after frozen DTU evidence

## One-sentence story

Known image preprocessing changes the camera matrix; CamCanon3R shows with
paired GT on VGGT and DUSt3R across ETH3D and held-out DTU that off-center
image coordinates cause reconstruction-camera drift even when RGB support is
preserved, then validates disagreement as a bounded failure detector and known-
affine canonicalization as an orientation-only repair.

## Claim -> mechanism -> evidence -> boundary

1. **Problem:** preprocessing is a camera operation, so reconstruction should
   be equivariant after exact camera correction and global gauge removal.
2. **Mechanism:** user and model pixel maps compose; off-center canvas position
   changes principal point, and view-dependent placement amplifies mismatch.
3. **Evidence:** 770 main GT records across 35 scenes, two models, two datasets,
   eleven transforms; severity/scope and support-preserving causal controls.
4. **Response:** held-out disagreement AUROC and one-pass canonical-camera
   repair, with native confidence, oracle, clean cost, point/depth, and compute
   controls.
5. **Boundary:** no deployed-prevalence claim, no universal confidence
   superiority, no generic geometry repair, no incomplete point bootstrap.

## Claim-evidence map

| Claim | Frozen evidence | Status and boundary |
|---|---|---|
| Exact preprocessing defines a 3D equivariance contract. | $K'=AK$, stored user/model maps, unit-tested composition and inverse mapping. | Supported; algebraic contract. |
| Independent off-center crops degrade GT cameras across models and datasets. | 75% rotation deltas: ETH3D 4.61°/4.52°, DTU 5.19°/2.96° for VGGT/DUSt3R; all CIs exclude zero. | Supported for tested scenes, views, models, and transforms. |
| Shared off-center crops form a second family; view dependence amplifies it. | Frozen 90/75/60% sweep; shared family crosses in all four model/dataset aggregates and independent crops are generally larger. | Supported at the registered family gate; no claim that every severity crosses. |
| Camera-canvas position alone can induce drift. | Center/shared-edge/independent-edge letterboxes preserve every source RGB pixel, scale, canvas, and padding count; independent deltas are 5.69°--13.97° across all four combinations. | Supported with disclosed post-ETH3D-mechanism registration chronology. |
| Disagreement detects held-out rotation failures. | DTU AUROC 0.998 [.990,1] for VGGT and 0.855 [.732,.949] for DUSt3R; risk--coverage, oracle/excess AURC, clustered intervals, cases hashes. | Supported at absolute 0.75 gate. |
| Disagreement universally beats native confidence. | Native AUROC 0.647 for VGGT and 0.855 for DUSt3R. | Fails: strong VGGT win, DUSt3R tie. Must remain explicit. |
| Known-affine canonicalization repairs camera orientation across datasets. | ETH3D recovery 0.966/0.558; DTU 0.941/0.767; zero measured identity cost for both models. | Supported for rotation. Both DTU lower CIs exceed 0.30. |
| Canonicalization repairs generic geometry. | Every ETH3D fill worsens depth; DTU VGGT completeness improves but accuracy confidence is weaker and DUSt3R point aggregates are incomplete. | Not supported; orientation-only claim. |
| Three-fill consensus beats one-pass repair. | Small VGGT improvement, DUSt3R tie, about 3x model compute. | Fails multi-model gate; retained as negative. |

## Final manuscript reverse outline

| Section | First-message role | Required evidence or contrast |
|---|---|---|
| Abstract | One systems variable causes measurable 3D failure. | 770 evaluations, cross-dataset deltas, support-preserving control, held-out AUROC, DTU repair, narrow boundary. |
| Introduction | A crop is a camera update, not generic corruption. | Exact contract, audit ambiguity, two-family result, causal control, detector/repair, four evidence-bounded contributions. |
| Related work | Distinguish complementary problem axes. | π³: permutation/reference symmetry; Cameras as Rays: learned ray distribution; PF-LRM: joint pose/shape systems evidence. No fake leaderboard. |
| Method | Remove known pixel maps and only the unavoidable 3D gauge. | $C_i=B_iA_i$, common-domain metrics, inverse warp, deterministic disagreement and tie-aware risk. |
| Experiments | Progress from replication to causal isolation and response. | Protocol -> cross-dataset main table -> severity -> support control -> repair -> held-out reliability -> compute. |
| Limitations | State exactly what the evidence does not establish. | Deployment prevalence, two-model/view scope, registration chronology, undefined DTU point directions, depth failure, native-score tie, license. |
| Conclusion | Turn the evidence into one practical decision. | Log/compose pixel maps, propagate induced cameras, canonicalize known mismatches before trusting geometry. |

## Three-ICLR comparison rule

- **π³:** both works define a symmetry and test camera/point/depth robustness;
  π³ changes input order/reference choice, CamCanon3R changes image coordinates.
- **Cameras as Rays:** it models pose ambiguity with sampled rays; CamCanon3R's
  disagreement is deterministic and earns only a held-out ranking claim.
- **PF-LRM:** it demonstrates broad joint pose/shape system evaluation while
  assuming known intrinsics; CamCanon3R audits the upstream preprocessing
  boundary that changes those intrinsics.

The working title remains audit-only. Rotation repair passes, but depth and
consensus do not justify adding “and Repairing” or claiming a general remedy.
