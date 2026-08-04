# Reviewer red-team

Updated: 2026-08-04
Checkpoint: second review from the frozen eight-page manuscript
Status: final evidence review; independent replication still desirable

## Recommendation

**ICLR-style estimate: central 7/10 (weak accept), plausible reviewer band
6--8. Confidence: 4/5.**

The honest floor has moved from the earlier 5--6 draft into the requested
6--8 band. The paper now has a precise new systems question, a causal result,
two substantially different reconstruction models, two geometry datasets,
held-out reliability, cross-dataset orientation repair, strong negative
boundaries, outcome-independent figures, compute, and unusually complete
provenance. A skeptical reviewer can still assign 6 because the intervention
suite is controlled rather than deployed, the detector covers transform-induced
failures rather than arbitrary real failures, and the repair is not a general
geometry method. An 8 is plausible for a reviewer who values correctness and
systems impact, but it is not the conservative prediction.

## Strongest acceptance case

1. **The overlooked variable is exact.** Preprocessing induces the camera
   update `K' = AK`; the paper turns a common implementation detail into a
   falsifiable equivariance contract instead of a generic corruption study.
2. **The main failure is replicated.** The full design has 770 GT records over
   13 ETH3D scenes, 22 held-out DTU scans, eleven transforms, VGGT, and DUSt3R.
   Independent 75% crop deltas are 4.61°/4.52° on ETH3D and 5.19°/2.96° on
   DTU; both off-center families cross the frozen gate on all model/dataset
   aggregates.
3. **The causal objection is unusually well controlled.** The edge-letterbox
   study preserves every source RGB byte, scale, canvas, and padding count yet
   produces 5.69°--13.97° rotation increases. Missing content is not necessary.
4. **The response mechanisms are useful but bounded.** Held-out disagreement
   AUROC is 0.998 and 0.855, and DTU rotation recovery is 0.941 and 0.767 at
   zero measured clean cost. Native-score ties, depth failure, and incomplete
   point aggregates are not hidden.
5. **The evidence is auditable.** Complete-design checks, scene-cluster
   intervals, source/config hashes, exact repeats, frozen qualitative scenes,
   explicit undefined values, compute boundaries, and a 1,615-entry bundle
   make post-hoc selection difficult.
6. **The paper reads as one argument.** Problem, mechanism, evidence, and
   boundary align across abstract, introduction, related work, experiments,
   limitations, and conclusion; there is no remaining TODO or future-tense
   promotion gate.

## Strongest rejection case

1. **Deployment relevance is inferred, not measured.** The interventions are
   realistic camera operations, but the paper does not quantify how often an
   actual product or user pipeline commits the audited mismatch.
2. **Breadth stops at two backbones and two datasets.** This is sufficient for
   a strong systems finding but weaker than a new reconstruction model tested
   across many tasks and domains.
3. **Held-out detection is intervention-bounded.** The 88 DTU cases per model
   come from the frozen transform set; AUROC does not establish calibration or
   detection of unrelated natural failure modes. DUSt3R only ties native AUROC.
4. **Repair is narrow.** Rotation recovers, but ETH3D depth worsens, DUSt3R DTU
   point aggregates are incomplete, and three-fill consensus fails. This is a
   safety response, not a general reconstruction improvement.
5. **The strongest control was designed after ETH3D mechanism results.** Its
   own outcomes and DTU GT remained prospective, and chronology is disclosed,
   but a fully independent preregistered replication would be stronger.
6. **Strict DTU metrics expose missingness.** One or two DUSt3R point directions
   are undefined under the frozen observation/20-mm filters. Refusing subset
   aggregation is correct, but it leaves cross-model DTU point evidence uneven.
7. **This is an audit paper, not a new backbone.** Reviewers who require a
   learned architectural advance may undervalue the practical systems result.

## Scorecard

| Dimension | Score | Evidence | Main residual risk |
|---|---:|---|---|
| Novelty | 7 | Exact preprocessing-equivariance contract and support-preserving causal intervention | Adjacent robustness reviewers may call the camera algebra obvious |
| Correctness | 8 | Exact maps, minimal gauge, paired GT, frozen thresholds, undefined-value discipline, post-signal amendments disclosed | DTU point protocol is deterministic but not a leaderboard protocol |
| Significance | 6 | Large camera drift across two leading model families; actionable preprocessing rule | No deployed incidence study or third backbone |
| Empirical strength | 7 | 35 scenes, 770 main records, 210 support controls, repair, reliability, compute, qualitative cases | Limited view counts and intervention-bounded failures |
| Presentation | 7 | Eight-page no-TODO narrative, four figures, five compact tables, explicit boundaries | Dense qualitative panels and no supplementary full-case atlas yet |
| Reproducibility | 8 | 157 tests, runbook, hashes, frozen cases, prediction digests, 833-file evidence bundle | Full data/weights remain external and DUSt3R license is non-commercial |

## Three-ICLR benchmark comparison

### π³: Permutation-Equivariant Visual Geometry Learning

π³ defines a symmetry, builds a representation around it, and validates pose,
point maps, depth, ordering, components, speed, and qualitative outcomes.
CamCanon3R now matches that evidence shape for a complementary symmetry:
known image coordinates rather than input permutation/reference frame. π³ has
the stronger learned-model and task-breadth contribution; CamCanon3R has the
stronger causal audit, chronology, negative-result, and artifact-provenance
story. The manuscript states complementarity and never claims π³ tested crops.

### Cameras as Rays: Pose Estimation via Ray Diffusion

Cameras as Rays motivates a distributed ray representation and samples learned
pose ambiguity. CamCanon3R's signal is deliberately narrower: deterministic
disagreement under equivalent preprocessing. The held-out AUROC/risk--coverage
plot now gives that signal a legitimate narrative role, while the DUSt3R tie
prevents it from being advertised as a universally superior uncertainty model.
The related-work contrast is precise and includes the practical per-view crop
and ray-grid example.

### PF-LRM: Pose-Free Large Reconstruction Model

PF-LRM supports a joint pose/shape system with broad cross-dataset and
robustness experiments while assuming known intrinsics. CamCanon3R audits the
upstream boundary that deterministically changes those intrinsics. It cannot
match PF-LRM's learned-model and object-dataset breadth, but it now compensates
with cross-dataset/cross-model GT, causal controls, held-out detection, repair,
compute, qualitative evidence, and stricter provenance. The paper does not
construct an invalid numerical leaderboard between the two tasks.

## Gate audit

- [x] DTU extraction, 146-file independent rehash, preparation, and exact
  22-scene x 11-variant design audited.
- [x] VGGT and DUSt3R complete separately; all camera/intrinsic records and all
  frozen point records retained.
- [x] Two-family/two-dataset mechanism gate passes unchanged.
- [x] Support-preserving control passes on all four model/dataset combinations.
- [x] Held-out detector exceeds AUROC 0.75 for both models; native tie retained.
- [x] Cross-dataset rotation repair passes for both models at zero clean cost;
  generic geometry and consensus remain negative.
- [x] Severity, risk--coverage, compute, cross-dataset, and frozen qualitative
  evidence are in the manuscript.
- [x] Every paper number is traceable to `artifacts/dtu_seed17/` or the earlier
  frozen ETH3D bundles.
- [x] No diagnostic/GT conflation, hidden scene selection, zero imputation,
  unsupported all-metric claim, or remaining TODO was found.

## What would make 8 more robust

The highest-value addition is one independently logged real preprocessing
pipeline or a third backbone evaluated with the existing audit. A second
independent replication of the support-preserving control would reduce the
post-ETH3D-registration concern. A supplement with all qualitative cases and a
minimal artifact-reproduction walkthrough would further help, but these are
improvements beyond the current weak-accept completion bar rather than blockers
for an honest 6--8 estimate.
