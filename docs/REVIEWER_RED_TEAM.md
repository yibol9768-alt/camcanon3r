# Reviewer red-team

Updated: 2026-08-04
Checkpoint: living pre-DTU review
Status: provisional review; not a final score

## Current recommendation

**ICLR-style estimate: 5--6 (borderline reject to weak accept).**

The paper already has a crisp and technically correct systems finding, unusually
strong provenance, two substantially different reconstruction systems, paired
ground-truth camera/depth/point evidence, mechanism controls, and honest
negative repair results. That is enough for a credible paper. It is not yet an
honest 6--8 case because all promoted accuracy and repair evidence comes from
one dataset, detector effectiveness has only been measured on a development
set, and the draft has no qualitative result figure or dedicated limitations
section.

The current score must not be rounded up because experiments are in progress.

## Strongest acceptance case

1. **The overlooked assumption is exact and consequential.** A crop is not a
   generic corruption: it induces the known camera update `K' = AK`. The paper
   converts this observation into a falsifiable equivariance contract.
2. **The failure is ground-truth, paired, and multi-model.** On 13 ETH3D scenes,
   independently shifted 75% crops increase median rotation by 4.61 degrees for
   VGGT and 4.52 degrees for DUSt3R, with matched translation, depth,
   principal-point, and completeness effects.
3. **The mechanism is better isolated than in a typical corruption study.**
   Retained-fraction sweeps, shared versus independent windows, center crops,
   letterbox, exact repeats, and per-model intervals separate off-center camera
   change from generic resampling and view-dependent amplification.
4. **Negative evidence is retained.** Canonicalization repairs rotation but
   worsens depth; consensus fails its multi-model promotion gate; undefined
   metrics are neither zero-filled nor silently dropped.
5. **Reproducibility is a real contribution.** Frozen protocols, resumable
   inference, source/config hashes, byte-exact identity audits, complete-design
   checks, and committed machine-readable evidence make the claims unusually
   auditable.

## Strongest rejection case

1. **One-dataset promotion is not broad enough.** ETH3D alone cannot establish
   the registered two-dataset mechanism hypothesis or cross-dataset repair.
2. **The detector is not yet a contribution.** ETH3D AUROC is promising
   development evidence, but the score was frozen after ETH3D outcomes were
   known. Without DTU held-out performance, the paper has an audit signal, not
   a validated failure detector.
3. **The repair contribution is narrow.** Rotation recovery is useful, but all
   registered fills worsen depth for both models and three-run consensus does
   not beat the analytic DUSt3R baseline. The title correctly avoids claiming a
   general repair method.
4. **The paper is visually under-evidenced.** There is no overview diagram,
   representative success/failure reconstruction figure, risk--coverage plot,
   or severity curve. Tables alone make the causal story harder to verify.
5. **Controlled crops may be dismissed as synthetic.** The camera intervention
   is causal and useful, but the paper still needs to connect it to realistic
   heterogeneous preprocessing pipelines without overstating prevalence.
6. **The six-page draft is dense but incomplete.** It lacks a compact
   compute/VRAM comparison and a final cross-dataset table. The TODO in failure
   detection makes the current draft non-submittable.

## Scorecard

| Dimension | Current | Evidence | Needed for 6--8 |
|---|---:|---|---|
| Novelty | 6 | Precise preprocessing-equivariance contract and affine-aware audit | Clarify practical prevalence and contrast with adjacent robustness/equivariance work |
| Correctness | 7 | Exact maps, minimal gauge, paired GT, controls, hashes, 123 tests | Complete DTU audit without protocol drift |
| Significance | 5 | Large failure on two major model families | Cross-dataset replication and a useful held-out detector or broader practical consequence |
| Empirical strength | 5 | 13 scenes, 286 mechanism evaluations, repair and development reliability | DTU 22 scenes x 11 variants x 2 models, held-out reliability, qualitative and compute views |
| Presentation | 5 | Clear claim boundaries, compact tables, and dedicated limitations | Overview/result figure and final no-TODO narrative |
| Reproducibility | 7 | Frozen protocols and committed evidence | Freeze DTU artifacts and final paper-number provenance |

## Mandatory completion gates

### P0: blocks an honest 6

- [ ] Complete and audit the exact 22-scene DTU preparation.
- [ ] Run VGGT and DUSt3R sequentially over all 11 frozen variants.
- [ ] Evaluate all 242 pose/intrinsic cases per model and the 88 frozen
  point-map cases per model.
- [ ] Recompute the two-family/two-dataset mechanism gate without changing
  thresholds or transform definitions.
- [ ] Build the four frozen DTU reliability cases per scene and report
  disagreement/native AUROC, risk--coverage, oracle AURC, excess AURC, and
  scene-cluster intervals for each model.
- [ ] Freeze representative qualitative successes and failures by a declared
  rule that does not select only favorable scenes.
- [ ] Remove every TODO after held-out results are frozen.
- [x] Add a dedicated limitations section covering intervention, evaluation,
  repair, detector, dataset/model, and license boundaries.

### P1: separates a 6 from a plausible 7

- [ ] Put ETH3D and DTU main outcomes into one compact cross-dataset table.
- [ ] Add a severity curve that shows center/shared/independent behavior for
  both models and both datasets.
- [ ] Add a held-out risk--coverage plot with native confidence and oracle.
- [ ] Report end-to-end compute, model-only compute, peak VRAM, and
  accuracy/recovery per model run.
- [ ] Present neutral, black, image-mean, native selection, consensus, and
  oracle repair in one ablation table, including the negative depth boundary.
- [ ] Add at least one realistic pipeline example or documented deployment
  scenario showing how per-view off-center preprocessing arises.

### P2: needed before claiming 8

- [ ] Demonstrate breadth beyond two backbones or two datasets, or provide a
  detector/repair result strong enough to compensate.
- [ ] Show that the central finding changes a practical design decision rather
  than only documenting a benchmark failure.
- [ ] Survive a second blinded red-team with no unresolved selection,
  alignment, leakage, or baseline objection.

## Comparison with the three ICLR writing benchmarks

### π³

π³ defines one symmetry, builds its representation around that symmetry, then
supports it with camera, point-map, depth, permutation robustness, component
ablation, speed, qualitative results, and explicit limitations. CamCanon3R is
already competitive in causal audit rigor, negative controls, and explicit
limitations, but it is behind in dataset/task breadth and visual evidence. DTU
and the severity plot are the minimum remaining structural match.

### Cameras as Rays

Cameras as Rays makes a single representation change easy to understand,
compares it with direct pose regression, sweeps 2--8 views, ablates ray
resolution, and visualizes sampled ambiguous modes. CamCanon3R has an equally
clean camera-level mechanism and a stronger registered causal intervention,
but its disagreement signal must be validated on held-out GT and visualized
through risk--coverage and representative cases before it can play the same
narrative role.

### PF-LRM

PF-LRM supports a systems claim with cross-dataset evaluation, variable-view
and mask-noise robustness, model/objective/pose-solver ablations, qualitative
reconstructions, and a concrete limitations list. CamCanon3R should not mimic
its leaderboard because the task differs. It must instead compensate with
cross-dataset, cross-model audit breadth; a complete baseline/compute table;
and explicit limits around missing pixels, synthetic interventions, model
licenses, and rotation-only repair.

## Final-review rule

After DTU artifacts and figures are frozen, rerun this review from the paper
alone before reading the authors' intended claim map. A 6 requires that the
main finding survives the second dataset and that every promoted sentence is
traceable to a frozen artifact. A 7 requires either held-out detector success
with useful coverage or a comparably strong practical result in addition to
the audit. An 8 is not presumed and should be assigned only if breadth,
impact, and presentation match the stronger ICLR benchmarks rather than merely
meeting the registered thresholds.
