# DTU point-evaluation implementation audit

Updated: 2026-08-04  
Status: frozen before any CamCanon3R DTU ground-truth outcome was computed

## Runtime failure-handling amendment (2026-08-05 Asia/Shanghai)

The frozen threshold and evaluated cases are unchanged. After the complete
VGGT summary had been written and the DUSt3R evaluator had written 55 of 242
case records, the first requested point-map metric for `scan12/identity`
retained no distances under the strict `< 20 mm` filter and the original
implementation stopped the task. This runtime error is an outcome signal and
is disclosed as such; it is not described as a preregistered amendment.

Before opening either model's aggregate GT summary or any held-out reliability
outcome, failure handling was made model-neutral and direction-specific. The
20 mm threshold is not relaxed, the case is not retried or dropped, and an
empty direction is never replaced by zero. Pose and intrinsics remain valid;
each point direction with no retained distance is recorded as undefined and
excluded only from bootstrap intervals requiring a complete scene set. Every
case reaching distance thresholding records accuracy and completeness counts
before and after filtering.
The same implementation applies to both models, all scenes, and all registered
point-map variants. The 55 already written DUSt3R records are retained because
none raised this condition and their numerical path is unchanged.

## Official source identity

The evaluator was checked against the official MATLAB files selected from DTU
`SampleSet.zip`.  These source bytes are preserved on `my5090` and selected by
`sampleset.json` with SHA-256
`c9417d72e0ee673b6531bdd3652797622d1a8184f3d1191400bb126270cb8d79`.

| Official file | SHA-256 |
| --- | --- |
| `PointCompareMain.m` | `bf8c6ba59e3a6b4aaf175a7ebdde3e8f87cb9ea78cf7ef387e5a12ba717e3e58` |
| `ComputeStat_web.m` | `4fb782bd65c9046aba2025792d44377efa6153e7df983e5683550544ca9158a9` |
| `reducePts_haa.m` | `e0e2592ccbef07581af8f3021fe071c9b1111f1e516ae937a7215ed4f1069d62` |

No surface scores, aggregate DTU metrics, or model outcomes were inspected
during this audit.

## Directional filter correspondence

The two official distance directions intentionally use different spatial
filters:

| Reported quantity | Official MATLAB path | CamCanon3R path |
| --- | --- | --- |
| Accuracy (prediction to STL) | `Ddata`, then `DataInMask`, then `< 20 mm` | predicted points inside `ObsMask` to the full STL tree, then `< 20 mm` |
| Completeness (STL to prediction) | `Dstl`, then `StlAbovePlane`, then `< 20 mm` | STL points above `Plane*.mat` to the full predicted-point tree, then `< 20 mm` |

The observation mask must **not** remove points from the tree queried for
completeness.  An earlier pre-result implementation applied it to both
directions; protocol `dtu-1.1` corrects that error.  A synthetic regression
test masks an exactly matching predicted/STL point and verifies that it is
excluded from accuracy but still yields zero completeness distance.

The official voxel lookup computes
`round((point - BB_min) / Res + 1)` in MATLAB one-based coordinates.  The
zero-based implementation `floor((point - BB_min) / Res + 0.5)` is equivalent
for in-bounds voxels.  The sign test `P' * [Qstl; 1] > 0` is preserved exactly.
`PointCompareMain.m` uses a 60 mm search cutoff internally, while
`ComputeStat_web.m` reports only distances below 20 mm; exact nearest-neighbor
queries followed by the same strict `< 20 mm` filter preserve the reported
set.

## Deliberate non-leaderboard differences

CamCanon3R evaluates point maps emitted by feed-forward camera systems, not an
official DTU MVS submission.  It therefore declares the following differences:

- predicted geometry is aligned with a camera-pose-only Sim(3); surface GT is
  never used to fit the gauge;
- 0.2 mm downsampling is deterministic voxel selection, whereas the official
  helper randomly orders points before radius suppression;
- predicted and above-plane target queries use deterministic 100,000-point
  caps for bounded repeatable cost;
- the frozen three-view mechanism sweep evaluates only its four registered
  confirmatory variants for point geometry; the separate two-variant repair
  sweep evaluates both identity and canonical crop so gap recovery is paired.

Accordingly, the paper must call these values the **CamCanon3R deterministic
DTU point-map metric**, not official DTU leaderboard scores.  For every case
that reaches distance evaluation, both direction counts, the observation-mask
count, alignment, cap, voxel size, and strict outlier threshold remain
machine-readable in the result.

The repair evaluator uses the identical directional filters and numerical
point-map implementation under protocol `dtu-repair-1.0`.  It additionally
binds every output to `configs/dtu_repair_protocol.json`, the base DTU and
qualitative protocols, and the audited canonical-input tree hash.  Repair
results never enter the eleven-variant mechanism summary.
