# Frozen qualitative evidence protocol

Status: frozen before DTU ground-truth extraction and evaluation

Qualitative figures must not be assembled by browsing the final errors and
keeping only persuasive scenes. The primary scene set is selected without
model predictions or ground-truth outcomes. From each frozen ordered scene
list, take indices `0`, `floor((n-1)/3)`, `floor(2(n-1)/3)`, and `n-1`.

This gives:

- ETH3D: `courtyard`, `kicker`, `playground`, and `terrains`;
- DTU: `scan1`, `scan15`, `scan48`, and `scan118`.

The same scenes are shown for VGGT and DUSt3R and for identity,
`asymmetric_crop_075`, and neutral-gray
`canonical_asymmetric_crop_075` when that registered repair exists. A selected
scene is never replaced because its result is weak, visually unattractive, or
undefined. Undefined metrics and degenerate geometry are displayed as such.

## Display contract

- Use the evaluator's camera-pose-only, orientation-preserving Sim(3);
  surface ground truth never fits the visualization.
- Use one declared axis range, point cap, point size, and colormap rule within
  each dataset figure. Do not tune these per scene or model.
- Show input support or validity masks next to repaired results so missing
  pixels are visible.
- Keep models in separate rows and report raw identity, corrupted, and repaired
  errors in the caption.
- Quantitative severity and risk--coverage plots use every registered scene and
  are not qualitative substitutes.

The machine-readable source of truth is
`configs/qualitative_protocol.json`. Any appendix of best/worst cases must be
clearly labeled diagnostic and cannot replace this fixed primary panel.
