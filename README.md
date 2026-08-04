# CamCanon3R

**A crop is a camera change.** CamCanon3R audits preprocessing-induced
geometry drift in feed-forward 3D reconstruction models and tests
known-affine canonicalization as a bounded response.

Modern models such as VGGT, DUSt3R, and MASt3R accept images after resizing,
cropping, padding, or aspect-ratio normalization. These operations are not
geometry-neutral: an image-space affine transform `A` changes the camera matrix
from `K` to `A @ K`. CamCanon3R asks whether model outputs obey this known
equivariance, measures failures in camera, depth, and point-map predictions,
and evaluates analytic and training-free consistency responses.

## Status

The frozen ETH3D study contains 13 scenes, 11 preprocessing variants, and 143
ground-truth evaluations per model. Independently shifted 75% crops increase
median rotation error by 4.61° for VGGT and 4.52° for DUSt3R; shared off-center
75% crops form a second, smaller failure family at 2.38° and 2.18°. Center and
letterbox controls stay below the registered mechanism threshold. Raw evidence
is committed in [`artifacts/eth3d_mechanism_seed17/`](artifacts/eth3d_mechanism_seed17/).

Known-affine neutral-gray canonicalization recovers 96.6% and 55.8% of the
paired rotation-error gap with zero measured identity clean cost. It worsens
depth for both models, and three-fill consensus fails its multi-model promotion
gate, so no generic geometry-repair claim is made. The complete positive and
negative evidence is in
[`artifacts/eth3d_repair_seed17/`](artifacts/eth3d_repair_seed17/).

The held-out 22-scene DTU study is complete. Independent 75% crops increase
paired rotation error by 5.19° for VGGT and 2.96° for DUSt3R; the shared
off-center family also crosses the frozen threshold at stronger severity on
both models. A support-preserving placement control keeps every source RGB
pixel, scale, canvas, and padding count fixed yet increases rotation by
5.69°--13.97° across all four model/dataset combinations. Missing support is
therefore not necessary for the camera drift.

On held-out DTU, cross-transform disagreement detects strict >2° rotation
failures with AUROC 0.998 for VGGT and 0.855 for DUSt3R. It strongly beats
VGGT native confidence but ties DUSt3R native AUROC, so universal superiority
is not claimed. Neutral-gray canonicalization recovers 94.1% and 76.7% of the
DTU rotation gap at zero measured clean cost. The complete lightweight DTU,
support-control, reliability, repair, compute, figure, and provenance evidence
is committed in [`artifacts/dtu_seed17/`](artifacts/dtu_seed17/); its bundle
contains 833 copied files and hashes 782 large predictions.

The no-TODO eight-page manuscript and second reviewer assessment are tracked in
[`paper/`](paper/) and [`docs/REVIEWER_RED_TEAM.md`](docs/REVIEWER_RED_TEAM.md).

Ground-truth experiments use only documented official datasets. See
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for URLs, hashes, formats,
and the raw-versus-undistorted ETH3D boundary.

## Core protocol

For each multi-view scene, compare the original input with controlled spatial
transforms whose coordinate affines are nonsingular. Crop-induced loss of
finite image support is recorded explicitly and is not claimed to be
invertible:

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
The exact my5090 severity and ETH3D commands are frozen in
[`docs/EXPERIMENT_RUNBOOK.md`](docs/EXPERIMENT_RUNBOOK.md).

The evolving 3DV 2027 manuscript uses the official author kit under
[`paper/`](paper/); its story and promotion gates are tracked in
[`docs/PAPER_STORY.md`](docs/PAPER_STORY.md).

For the split control/execution deployment, where Codex works on `vircs` and
GPU experiments remain on `my5090`, start with
[`docs/VIRCS_HANDOFF.md`](docs/VIRCS_HANDOFF.md).

Prepare deterministic pilot variants with exact image-space transforms:

```bash
camcanon3r prepare-scene /path/to/scene /path/to/prepared --max-views 8 --seed 17
```

The output contains one folder per variant and a `manifest.json` recording every
source image, target image, resolution, interpolation rule, seed, and 3x3
source-to-target pixel matrix.

Known transforms can be inverse-warped onto their original camera canvas for
the analytic repair baseline. The repaired manifest stores identity camera
coordinates, the original affine, fill policy, validity mask, and visible
support fraction:

```bash
PYTHONPATH=src python scripts/canonicalize_variant.py \
  /path/to/prepared/asymmetric_crop_075 \
  /path/to/repaired/canonical_asymmetric_crop_075
```

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
