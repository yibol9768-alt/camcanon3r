# Frozen three-scene DUSt3R diagnostic

Date: 2026-08-04  
Views per scene: 4  
Seed: 17  
Inference checkout: `5af64f1`  
Summary checkout: `8b32408`  
Model: official DUSt3R ViT-L/512 DPT checkpoint

This confirmatory diagnostic independently tests whether the preprocessing
non-equivariance observed with VGGT also appears in DUSt3R. It compares every
transformed run with the same scene's identity prediction after exact affine
bookkeeping. It is not a ground-truth accuracy experiment.

| Scene | Transform | Median rotation | Median translation direction | Mean aligned depth AbsRel |
|---|---|---:|---:|---:|
| kitchen | asymmetric crop, 75% | 6.335° | 16.254° | 12.942% |
| llff_fern | asymmetric crop, 75% | 8.831° | 56.391° | 7.314% |
| room | asymmetric crop, 75% | 5.741° | 83.400° | 3.987% |
| kitchen | center crop, 75% | 1.731° | 15.098° | 10.359% |
| llff_fern | center crop, 75% | 2.298° | undefined | 5.679% |
| room | center crop, 75% | 5.566° | 84.054° | 3.064% |
| kitchen | square letterbox | 0.304° | 0.768° | 4.035% |
| llff_fern | square letterbox | 0.288° | 17.598° | 2.019% |
| room | square letterbox | 1.608° | 11.159° | 1.839% |

The asymmetric crop exceeds the frozen 2° rotation threshold in all three
scenes. Its median-of-scene median rotation disagreement is 6.335°, with a
descriptive three-scene bootstrap interval of 5.741--8.831°. Center crop
exceeds the threshold in two scenes, while letterbox exceeds it in none. This
independently confirms the narrow multi-model claim that view-dependent crop
preprocessing can change recovered geometry in both VGGT and DUSt3R.

The `llff_fern` center-crop translation statistic is intentionally undefined.
DUSt3R returned identical zero camera centers for all four transformed views,
so every candidate baseline has zero length and no direction. The summarizer
records one undefined scene and excludes translation from that variant's
scene bootstrap; it does not drop the scene, substitute zero, or pool it with
other metrics. The pose collapse is itself a qualitative failure, but its
accuracy consequence still requires ground truth.

The batch reused one model load across all 12 runs. The cold load took 32.45 s;
after the first warm-up run, pairwise inference took 1.04--1.19 s and global
alignment took 3.68--4.12 s per case. Peak allocated VRAM was approximately
2.78 GB. Machine-readable comparisons, prediction metadata, and the full
summary are stored in
[`artifacts/dust3r_pilot_seed17`](../artifacts/dust3r_pilot_seed17).

This result promotes cross-model diagnostic generality only. ETH3D or DTU
ground-truth accuracy, held-out failure detection, and repair remain pending
claim gates.
