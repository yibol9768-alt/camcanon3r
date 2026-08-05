# Canonical orbit projection runbook

This runbook is for the `fancy/orbit-projection` development branch. The
frozen audit baseline remains commit
`e03cbb34cad2a87ed9f12788b93dba3901ee9cd5`. The method protocol is
`configs/orbit_projection_protocol.json`. Paper editing is outside this phase.

## Execution boundary

Run code editing, tests, protocol review, commits, and GitHub pushes on vircs.
Run image preparation, model inference, projection over large prediction
archives, and GT evaluation under `/opt/camcanon3r` on my5090. Do not enable a
global proxy or TUN. Orbit jobs require no download proxy.

Every long command must use `scripts/start_my5090_background_job.sh`. Never
leave model inference attached to the SSH session. Run at most one model sweep
at a time.

## Frozen design

Each scene starts from the neutral-gray
`canonical_asymmetric_crop_075` repair. Nine members embed the complete repaired
RGB image and its validity mask on an identically sized 1.25x canvas without
resampling. The center and four inverse placement pairs are fixed in the
protocol. The audit verifies every decoded RGB and mask byte, the fill region,
the registered translation, file hashes, and complete scene counts.

The projection consumes all nine camera predictions. It averages inverse
placement pairs in the gauge-invariant relative-rotation graph, uses Tukey
weights at the group level, and runs complete-graph SO(3) synchronization. A
separate uniform projection, orbit medoid, native-confidence selector, and GT
oracle are evaluated from the same predictions.

## Local validation and sync

On vircs:

```bash
cd /root/Desktop/lyb/camcanon3r
.venv/bin/ruff check src tests scripts
.venv/bin/python -m pytest -q
git diff --check
git push
./scripts/sync_my5090_checkout.sh
```

The sync helper must confirm clean, non-overlapping worktrees. Keep the user
artifact `CamCanon3R_Overleaf_2026-08-05.zip` untracked and untouched.

## ETH3D development stage

Prepare and audit the complete orbit:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-OrbitPrep-ETH3D \
  'cd /opt/camcanon3r; ./scripts/run_orbit_preparation.sh eth3d \
  > results/orbit/eth3d_preparation.log 2>&1'
```

After the task exits with code zero, run VGGT and DUSt3R sequentially:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-OrbitVGGT-ETH3D \
  'cd /opt/camcanon3r; ./scripts/run_orbit_inference.sh vggt eth3d \
  > results/orbit/eth3d_vggt_inference.log 2>&1'

./scripts/start_my5090_background_job.sh CamCanon3R-OrbitDUSt3R-ETH3D \
  'cd /opt/camcanon3r; ./scripts/run_orbit_inference.sh dust3r eth3d \
  > results/orbit/eth3d_dust3r_inference.log 2>&1'
```

Project and evaluate only after both complete:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-OrbitEvalVGGT-ETH3D \
  'cd /opt/camcanon3r; ./scripts/run_orbit_projection_evaluation.sh vggt eth3d \
  > results/orbit/eth3d_vggt_evaluation.log 2>&1'

./scripts/start_my5090_background_job.sh CamCanon3R-OrbitEvalDUSt3R-ETH3D \
  'cd /opt/camcanon3r; ./scripts/run_orbit_projection_evaluation.sh dust3r eth3d \
  > results/orbit/eth3d_dust3r_evaluation.log 2>&1'
```

ETH3D is development data for this method because its earlier mechanism and
repair outcomes were known. Retain failed baselines and all nine member errors.

## DTU method-outcome freeze and confirmation

Do not prepare or evaluate the DTU orbit until the ETH3D ablation has selected
one implementation without DTU projected GT inspection. Record the selected
commit and config SHA-256 in the protocol, commit, push, and resync first.

Then use the same four-stage order:

```bash
./scripts/start_my5090_background_job.sh CamCanon3R-OrbitPrep-DTU \
  'cd /opt/camcanon3r; ./scripts/run_orbit_preparation.sh dtu \
  > results/orbit/dtu_preparation.log 2>&1'

./scripts/start_my5090_background_job.sh CamCanon3R-OrbitVGGT-DTU \
  'cd /opt/camcanon3r; ./scripts/run_orbit_inference.sh vggt dtu \
  > results/orbit/dtu_vggt_inference.log 2>&1'

./scripts/start_my5090_background_job.sh CamCanon3R-OrbitDUSt3R-DTU \
  'cd /opt/camcanon3r; ./scripts/run_orbit_inference.sh dust3r dtu \
  > results/orbit/dtu_dust3r_inference.log 2>&1'

./scripts/start_my5090_background_job.sh CamCanon3R-OrbitEvalVGGT-DTU \
  'cd /opt/camcanon3r; ./scripts/run_orbit_projection_evaluation.sh vggt dtu \
  > results/orbit/dtu_vggt_evaluation.log 2>&1'

./scripts/start_my5090_background_job.sh CamCanon3R-OrbitEvalDUSt3R-DTU \
  'cd /opt/camcanon3r; ./scripts/run_orbit_projection_evaluation.sh dust3r dtu \
  > results/orbit/dtu_dust3r_evaluation.log 2>&1'
```

The earlier DTU baseline outcomes were already known, so this is held out only
with respect to the new projected outputs after the implementation freeze. It
must not be described as a fully prospectively selected benchmark.

## Promotion and conditional student

The multi-run method must pass every model-dataset gate in the protocol:

1. at least 15 percent residual rotation-gap reduction over one-pass repair;
2. no median degradation greater than 0.1 degrees;
3. robust projection beats or ties uniform projection and the orbit medoid;
4. complete scene-cluster bootstrap and compute accounting;
5. no GT, uncropped pixels, changed retry, or hidden undefined value.

Only a full multi-run pass triggers development of the affine-aware graph
student. The student must retain 80 percent of the teacher gain using one
backbone forward pass. A failed student remains a negative result and cannot
change the multi-run gate.
