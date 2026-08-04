# ETH3D exploratory reliability study, seed 17

This directory freezes the development-only reliability analysis on all 13
ETH3D training scenes. VGGT and DUSt3R remain separate. Each model contributes
52 complete cases: identity, center crop at 75% retention, independently
shifted crop at 75%, and square letterbox. ETH3D outcomes were inspected before
the reliability score was frozen, so these results validate the pipeline and
motivate the held-out test; they do not promote a detector claim.

## Rotation failure detection

The primary error is GT relative-rotation error, with failure defined strictly
as error greater than 2 degrees.

| Model | Score | Failures | AUROC (95% CI) | Excess AURC |
|---|---|---:|---:|---:|
| VGGT | Cross-transform disagreement | 15/52 | 0.924 [0.816, 1.000] | 0.318 |
| VGGT | Native uncertainty | 15/52 | 0.665 [0.477, 0.865] | 0.709 |
| DUSt3R | Cross-transform disagreement | 23/52 | 0.908 [0.802, 0.973] | 0.210 |
| DUSt3R | Native uncertainty | 23/52 | 0.597 [0.414, 0.795] | 0.681 |

Disagreement is stronger than native uncertainty for both models in this
exploratory matrix, including lower excess AURC. The frozen detector gate is
not evaluated here: DTU is the held-out promotion dataset.

## Secondary depth failure detection

The secondary error is GT depth AbsRel, with failure defined strictly as error
greater than 0.05.

| Model | Score | Failures | AUROC (95% CI) | Excess AURC |
|---|---|---:|---:|---:|
| VGGT | Cross-transform disagreement | 7/52 | 0.900 [0.785, 0.986] | 0.00234 |
| VGGT | Native uncertainty | 7/52 | 0.679 [0.477, 0.844] | 0.00606 |
| DUSt3R | Cross-transform disagreement | 12/52 | 0.797 [0.551, 1.000] | 0.00465 |
| DUSt3R | Native uncertainty | 12/52 | 0.694 [0.309, 0.997] | 0.01182 |

The DUSt3R depth interval is wide. No depth detector is promoted from these
development results.

## Protocol and integrity

- Cross-transform uncertainty is each candidate's median disagreement with
  the other three frozen transforms. Identity is not assigned zero.
- Native uncertainty is negative median model confidence. Ground truth never
  enters either score.
- Intervals use 10,000 scene-cluster bootstrap replicates, 95% confidence, and
  seed 17. AUROC uses average ranks and risk--coverage retains complete tie
  blocks.
- Case schema 1.1 stores source roots once and case files relative to those
  roots. Every statistics report records the SHA-256 of its case file.
- The exact staged tool archive came from commit
  `fa470bec172870fb564d2831f081e6f145e8b33b`. The enclosing 11-variant
  mechanism configuration SHA-256 is
  `2219be64a9c0e829ce619fe8aa431f53aad8c7ef7695a5bd570d48054187100e`.
- `SHA256SUMS` covers all ten JSON evidence files.
