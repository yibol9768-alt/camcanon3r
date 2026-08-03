# CamCanon3R

**A crop is a camera change.** CamCanon3R audits and repairs
preprocessing-induced geometry drift in feed-forward 3D reconstruction models.

Modern models such as VGGT, DUSt3R, and MASt3R accept images after resizing,
cropping, padding, or aspect-ratio normalization. These operations are not
geometry-neutral: an image-space affine transform `A` changes the camera matrix
from `K` to `A @ K`. CamCanon3R asks whether model outputs obey this known
equivariance, measures failures in camera, depth, and point-map predictions,
and develops a training-free consistency repair.

## Status

The repository currently contains the frozen geometry protocol and a 72-hour
kill-test contract. No performance claim is made before the kill-test passes.

## Core protocol

For each multi-view scene, compare the original input with controlled,
invertible image transforms:

- center and asymmetric crop-resize;
- isotropic and anisotropic resize;
- letterbox padding;
- mixed transforms across views.

The transformed camera must satisfy `K' = A @ K`. Predictions are mapped back
to the original image coordinates and compared after the minimal permitted
gauge alignment. A model is reliable only when its geometry remains consistent
and its confidence predicts residual failure.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

See [`docs/RESEARCH_CONTRACT.md`](docs/RESEARCH_CONTRACT.md) for hypotheses,
baselines, metrics, stop criteria, and the claim-evidence contract.

