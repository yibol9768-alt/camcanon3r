# Frozen pilot result: VGGT room, four views, seed 17

Date: 2026-08-03  
Code commit: `50e5b96`  
VGGT commit: `a288dd0f14786c93483e45524328726ab7b1b4ce`

This is a positive **non-equivariance** result on one official VGGT example
scene. It is not yet evidence that a transformed run is less accurate against
ground-truth geometry. That stronger claim remains blocked on multi-scene,
multi-dataset ground-truth evaluation.

## Result

All comparisons use the identity prediction as the reference. Camera metrics
compare all six relative-pose pairs and therefore cancel the arbitrary world
frame and translation scale. Depth is compared on common source-image support
after the logged protocol and VGGT preprocessing affines are inverted, followed
by one scene-level median scale alignment.

| Candidate | Median rotation | Median translation direction | Mean aligned depth AbsRel |
|---|---:|---:|---:|
| identity repeat | 0.000° | 0.0000004° | 0.0000% |
| center crop, 75% | 3.876° | 18.002° | 8.666% |
| asymmetric crop, 75% | 6.297° | 32.283° | 9.939% |
| square letterbox | 1.756° | 9.227° | 4.153% |

The repeat control is effectively exact, so the crop signals are not explained
by ordinary repeated-inference noise. Both crop variants pass the frozen pilot
threshold of more than 2° median relative-pose disagreement. The direction
therefore continues to ground-truth confirmation rather than being killed.

## Runtime envelope

The four-view crop tensor is `4 x 3 x 392 x 518`. Peak allocated VRAM is
5,868,528,640 bytes for identity/crop and 6,218,940,416 bytes for the larger
letterbox tensor. A cold 5 GB weight load took 72--77 seconds on the observed
WSL filesystem; warm loads took 6.6--7.1 seconds. Inference ranged from 0.42 to
0.48 seconds after warm-up, so load and warm-up must be reported separately.

## Audit boundary

- The protocol stores the intervention affine, VGGT's internal affine, and
  their source-to-tensor composition for every image.
- The first official room image is 512x384 while the other three are
  1280x960; this is logged rather than silently treated as equal resolution.
- Depth comparison uses only common visible pixels: about 56% for 75% crops
  and 100% for the letterbox run.
- No confidence, AUROC, repair, or ground-truth claim is made from this table.

The machine-readable summary is
[`artifacts/pilot_room_seed17_summary.json`](../artifacts/pilot_room_seed17_summary.json).
