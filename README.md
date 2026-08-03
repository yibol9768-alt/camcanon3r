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

The frozen three-scene VGGT diagnostic found a consistent asymmetric-crop
non-equivariance signal, with an effectively exact identity-repeat control.
See [`docs/MULTISCENE_RESULTS.md`](docs/MULTISCENE_RESULTS.md) and its committed
machine-readable evidence. This is not yet a ground-truth accuracy or
multi-model claim.

Ground-truth experiments use only documented official datasets. See
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for URLs, hashes, formats,
and the raw-versus-undistorted ETH3D boundary.

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

The evolving 3DV 2027 manuscript uses the official author kit under
[`paper/`](paper/); its story and promotion gates are tracked in
[`docs/PAPER_STORY.md`](docs/PAPER_STORY.md).

Prepare deterministic pilot variants with exact image-space transforms:

```bash
camcanon3r prepare-scene /path/to/scene /path/to/prepared --max-views 8 --seed 17
```

The output contains one folder per variant and a `manifest.json` recording every
source image, target image, resolution, interpolation rule, seed, and 3x3
source-to-target pixel matrix.

### Process-scoped downloads on my5090

Large external artifacts may be downloaded with the machine-local wrapper:

```bash
./scripts/with_download_proxy.sh huggingface-cli download OWNER/REPOSITORY
```

The wrapper starts a triggerless Windows task, exposes its WSL-adapter endpoint
only to the supplied command through environment variables, and stops the task
when that command exits. It does not change the Windows or WSL system proxy and
does not use TUN. The Mac-derived configuration and all provider credentials
remain in a machine-local private directory and are never committed. A local
Mihomo backend is also available as an explicit fallback.
