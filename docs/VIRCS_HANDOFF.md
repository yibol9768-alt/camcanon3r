# CamCanon3R handoff: vircs controls, my5090 executes

Updated: 2026-08-04 (Asia/Shanghai)

Live execution note (2026-08-05 Asia/Shanghai): DTU extraction, independent
146-file rehash, preparation, both 242-case inference sweeps, and the VGGT GT
evaluation are complete. The DUSt3R GT evaluator stopped after 55 records when
`scan12/identity` retained no point distance below the frozen 20 mm threshold.
Before aggregate GT summaries or held-out reliability outcomes were opened, a
model-neutral amendment was added to preserve pose/intrinsics and mark only an
empty point direction undefined, without changing the threshold, retrying, or
dropping a case. Resume the audited task only after this change passes tests,
is pushed, and the `my5090` checkout is safely fast-forwarded.

## Mission and completion bar

Continue CamCanon3R as a 3DV paper project.  The work is not complete merely
because the code runs or the current draft compiles.  Stop only after the
paper, evidence, reproducibility package, and reviewer red-team jointly support
an estimated ICLR-style score in the 6--8 range.  Report uncertainty honestly;
never promote a claim beyond the committed evidence.

The present paper is a seven-page working draft, not a finished submission.
Its Related Work is calibrated against the frozen three-paper ICLR comparison;
matched ETH3D mechanism, repair, and development-reliability evidence is
integrated; and a vector method overview plus dedicated limitations section are
present.  The remaining paper TODO is deliberately reserved for held-out DTU
reliability.  The current red-team estimate remains 5--6, not the requested
6--8, because cross-dataset GT, held-out detection, and result figures are not
yet complete.

## Two-machine contract

| Responsibility | Machine | Canonical path |
| --- | --- | --- |
| Codex session, code editing, tests, paper writing, Git commits/pushes | `vircs` (`sivenfuu`) | `/root/Desktop/lyb/camcanon3r` |
| CUDA inference, weights, datasets, full experiment outputs | `my5090` WSL Ubuntu | `/opt/camcanon3r` |
| Large ETH3D archives and extracted data | `my5090` Windows E: via WSL | `/mnt/e/camcanon3r-data` |
| Code source of truth | GitHub | `yibol9768-alt/camcanon3r`, branch `main` |

Do not copy model weights or full datasets onto `vircs`.  It has no visible
NVIDIA GPU.  Its 48 GB root filesystem had only about 1.3 GB free during the
initial check and about 16 GB at final verification after space was reclaimed
outside this migration; treat that capacity as volatile.  Do not delete code,
`/root/.claude`, or remote-development data to make space.  If cleanup becomes
necessary, obtain explicit user approval and use a rebuildable-cache whitelist.

## Start here on vircs

```bash
cd /root/Desktop/lyb/camcanon3r
git status --short --branch
git pull --ff-only origin main
./scripts/verify_vircs.sh
```

If the CPU environment has not been created:

```bash
./scripts/setup_vircs.sh
```

The setup script refuses to proceed when less than 700 MB is free.  It installs
only the small project/dev dependencies and runs the unit tests; it does not
install VGGT, DUSt3R, CUDA, or datasets on `vircs`.

## Run commands on my5090 from vircs

The Windows SSH server adds a `cmd.exe` parsing hop.  Always use the repository
wrapper so pipes, redirects, and shell operators reach WSL intact:

```bash
./scripts/vircs_remote_my5090.sh 'hostname; nvidia-smi'

./scripts/vircs_remote_my5090.sh '
  cd /opt/camcanon3r
  git status --short --branch
  pgrep -af download_archives.py || true
  tail -n 30 /mnt/e/camcanon3r-data/eth3d_archives/download.log || true
'
```

After pushing code from `vircs`, update the execution checkout without
discarding remote work:

```bash
git push origin main
./scripts/sync_my5090_checkout.sh
```

The sync script aborts if tracked changes exist on `my5090` and uses
the process-scoped download proxy for `git fetch`, followed by a fast-forward-
only merge.  It never resets or force-checks out files.

## Network rule

- The stable control path is `vircs -> my5090` over the existing SSH alias and
  dedicated key. Keep long runs detached on `my5090`; loss of the controller
  session must not kill an experiment. On this Windows OpenSSH-to-WSL topology,
  Linux-only `tmux` or `nohup` does not keep the distribution alive after the
  final Windows-side WSL client exits. Launch long commands with
  `scripts/start_my5090_background_job.sh`, which uses a triggerless Windows
  Scheduled Task with concurrent starts disabled. Five fresh control probes
  succeeded at handoff, with new-connection latency between about 2.9 and 5.2
  seconds.
- Git commits are pushed from `vircs` to GitHub, then fast-forwarded on
  `my5090`.  Commit lightweight summaries and provenance, not weights, raw
  datasets, or large prediction archives.
- Hugging Face and other large downloads run on `my5090` only, through
  `./scripts/with_download_proxy.sh COMMAND` or the frozen download launcher.
  The proxy is process-scoped: never enable a Windows/WSL global proxy and
  never enable TUN.
- Do not copy the Mac proxy configuration, provider credentials, private keys,
  `.env` files, or `.private-download-proxy/` into Git or `vircs`.

## Current frozen state

- Control-plane baseline immediately before this update:
  `e6fcaff44c83511387479a36c7be54d224b782e7` plus the extraction-audit
  commit that contains this text. `vircs` is clean and passes 154 CPU tests.
  While the DTU selection extractor owns the download proxy and data tree, the
  formal `my5090` checkout is intentionally left clean at `4029913`; it passes
  its then-current 123 tests.  Fast-forward it only after extraction exits.
- VGGT weights already reside on `my5090`.
- DUSt3R weights reside at
  `/opt/camcanon3r/checkpoints/dust3r-512-dpt`; the previously verified large
  weight SHA-256 is
  `7c300a89534113436bde52732d3151212bcbd90f0aa3c8d1496f86d84bfe4b42`.
- All 15 frozen ETH3D archives are downloaded and locally hashed under
  `/mnt/e/camcanon3r-data/eth3d_archives` (about 16 GB).  The strict extraction
  audit covers all 13 training scenes, four deterministic views per scene, 234
  selected source files, exact paths/sizes/SHA-256 values, and no extras.
  There is no active `download_archives.py` process.  Do not restart the
  downloader unless the completed report itself fails a fresh audit.
- Raw ETH3D preparation contains 13 scenes x 4 variants x 4 views: 208 PNGs
  plus 13 manifests under `data/eth3d_training/raw` (about 4.6 GB).  Its strict
  audit verified every image, affine, view order, and nested scene ID.  Mixed
  COLMAP camera IDs in `electro`, `facade`, and `terrace` are intentionally
  retained and evaluated per view.
- Full raw inference and protocol-2.1 GT evaluation are complete for both
  models: 52/52 VGGT cases and 52/52 DUSt3R cases, with all pose, intrinsic,
  depth, point-accuracy, and point-completeness metrics available.  Prediction
  roots are about 772 MB (VGGT) and 724 MB (DUSt3R); each result root is about
  568 KB.  The four Windows tasks `CamCanon3R-ETH3DRawVGGT`,
  `CamCanon3R-ETH3DRawVGGTEval`, `CamCanon3R-ETH3DRawDUSt3R`, and
  `CamCanon3R-ETH3DRawDUSt3REval` are Ready and last exited successfully.
- Lightweight exact summaries are committed at
  `artifacts/eth3d_vggt_raw_seed17/summary.json` and
  `artifacts/eth3d_dust3r_raw_seed17/summary.json`.  Each contains all 56
  scene-bootstrap metric triplets and source/prediction digests; both were
  programmatically matched against the full my5090 summary before commit.
- The decisive matched result is the view-dependent 75% crop.  Its paired
  rotation deltas are 4.61 degrees for VGGT (95% CI [3.13, 6.17]) and 4.52
  degrees for DUSt3R ([2.13, 5.72]); translation, depth, principal-point, and
  completeness intervals also exclude zero for both.  DUSt3R point accuracy
  does not exclude zero.  Letterbox depth and point intervals include zero for
  both models.  Do not broaden this to an all-transform or all-metric claim.
- The exact 11-variant severity/scope sweep is complete for 13 ETH3D scenes and
  both models (286 evaluations). Independent off-center rotation degradation
  is monotone and exceeds the frozen threshold at 75% and 60%; the shared
  off-center family also crosses, while center crop and letterbox remain
  controls. Machine-readable summaries and the combined mechanism analysis are
  committed under `artifacts/eth3d_mechanism_seed17/`.
- The three-fill repair study and exact identity repeats are frozen under
  `artifacts/eth3d_repair_seed17/`. Neutral-gray rotation recovery is 0.966 for
  VGGT and 0.558 for DUSt3R with zero measured clean cost. Every registered
  fill worsens depth for both models, and consensus fails its multi-model
  promotion gate. `ablation_summary.json` binds all eight input reports.
- ETH3D reliability is development-only. Rotation disagreement AUROC is 0.924
  for VGGT and 0.908 for DUSt3R, versus 0.665 and 0.597 for native confidence.
  These scores were visible during development and do not promote a detector;
  the frozen DTU evaluation remains the only held-out gate.
- DTU acquisition is active under the single Windows task
  `CamCanon3R-DTUSelectionExtract`. At this edit, SampleSet is complete 58/58,
  Rectified is in progress at 56/66, and Points has not started. The
  extractor uses only the process-scoped proxy, is resumable, and now retries
  truncated HTTP 206 bodies. Do not launch a duplicate, sync the execution
  checkout, inspect GT outcomes, or start GPU work while this task is running.
- DTU inference/evaluation wrappers are frozen on `vircs`: preparation requires
  22 scenes, 11 variants, and 726 PNGs; model sweeps are sequential; schema-1.2
  outputs bind input hashes and full timing; resume validates CRC and affines;
  evaluation performs audit-only prediction validation; and qualitative scene
  selection was frozen before outcomes.
- A separate post-extraction auditor must rehash the exact 146 selected DTU
  files against all three frozen selections and complete reports before
  preparation. It rejects content, report, membership, order, or extra-file
  drift, writes one tree digest, and is included in the final evidence bundle.
- The confirmatory paper prose now records the frozen DTU split, exact three
  views and lighting condition, eleven-transform matrix, four point-map
  variants, and the non-leaderboard boundary of the deterministic point-map
  metric. The DTU benchmark's primary CVPR 2014 citation is included. No DTU
  outcome was added to the draft before the held-out gate.
- The previously missing DTU canonical-control chain is now frozen separately:
  neutral-gray preparation requires 22 scenes, two variants, 132 images and
  132 masks; each model runs 44 predictions; identity repeats, compute, and
  prediction pairs are audited; both variants receive point GT; and the final
  paired report uses the unchanged 30% recovery and 2% clean-cost gates. This
  chain must not be mixed into the eleven-variant mechanism summary.
- The primary qualitative renderer is implemented and synthetic-end-to-end
  checked over all 24 panels. Its protocol freezes the selected scenes, first
  target-camera projection, 25,000 points per view, 320 x 240 z-buffer,
  camera-baseline depth normalization, viridis range, and canonical mask
  insets. Real ETH3D/DTU figures remain pending the formal DTU results.
- Compute accounting now has one provenance-bound table source. DTU schema-1.2
  model-only/end-to-end timing, model load, and VRAM stay separate; legacy
  ETH3D end-to-end time remains explicitly unavailable. DTU inverse-warp
  preparation atomically checkpoints its own decode/warp/write time and refuses
  an unaccounted resumed output.
- The final claim auditor now distinguishes a complete negative result from a
  promoted claim. It reopens and hashes held-out cases, enforces the frozen
  mechanism/reliability/repair designs, and reports five gates without turning
  them into an automated reviewer score.
- A separate support-preserving coordinate control is now frozen before DTU
  GT inspection. Symmetric, shared-edge, and independent-edge letterboxes keep
  every source RGB pixel, source scale, square canvas, and black-padding count
  identical. Strict audits bind the symmetric anchor to the main preparation;
  the primary two-degree gate remains separate from the eleven-variant matrix.
  It was registered after the existing ETH3D mechanism results but before any
  support-control outcome, and must be described with that exact chronology.
- The final evidence-bundle manifest now copies 782 lightweight GT evaluation
  records: the original 572 DTU mechanism/repair records plus 210 registered
  ETH3D/DTU support-control records. It hashes the corresponding 782 large
  predictions without copying them into Git. Bundle writes remain atomic and
  safely resumable.
- Latest idle GPU check after all jobs: 0% utilization and 1336 MiB baseline
  memory.  Live sizes were 4.7 GB for VGGT weights and 2.2 GB for DUSt3R
  weights.

Re-verify every item live before reporting it as current.  Machine state and
download progress are allowed to drift after this document is committed.

## Safe experiment sequence

1. Run `./scripts/verify_vircs.sh` and record both commit hashes.
2. If hashes differ, inspect both worktrees; push from `vircs`, then use the
   fast-forward sync script.  Never overwrite uncommitted work.
3. On `my5090`, check `nvidia-smi` process list and utilization.  Do not start
   VGGT or DUSt3R while another project owns the GPU.
4. Check the Windows scheduled task and Linux process/report for any active
   DTU or ETH3D transfer before touching a download. Never infer failure from a
   long no-output interval inside a remote ZIP member.
5. Follow `docs/EXPERIMENT_RUNBOOK.md` exactly. Use the Windows-owned background
   launcher for long commands, use resumable runners, and keep
   raw, transformed, repaired, and clean-control outputs distinct.
6. Bring only small machine-readable summaries, plots, provenance, and paper
   evidence back into Git.  Run the full CPU test suite before every push.
7. Update the paper only after claim gates in `docs/RESEARCH_CONTRACT.md`,
   `docs/PAPER_STORY.md`, and `docs/RELIABILITY_PROTOCOL.md` are satisfied.

## Immediate research queue

1. Let the existing DTU selection task finish exactly once; verify all three
   reports and the selected tree before syncing the `my5090` checkout.
2. Fast-forward `my5090`, run 154/154 tests, then complete the independent
   146-file extraction audit and execute/audit DTU preparation. Start no GPU
   task until two idle checks and no foreign owner.
3. Run VGGT then DUSt3R over the exact 22 x 11 design, retaining schema-1.2
   input hashes, compute/VRAM metadata, and audit-only resumability.
4. Evaluate all pose/intrinsic cases and the four point-map variants, freeze
   compact artifacts, then open the unchanged held-out reliability gate.
5. Run the separate two-variant canonical-control chain for both models and
   freeze DTU gap-recovery reports without changing the selected fill policy.
6. Render the frozen severity, held-out risk--coverage, repair-ablation,
   cross-dataset, compute, and outcome-independent qualitative evidence.
7. Run the support-preserving letterbox control over both models and datasets,
   then freeze its paired cross-dataset gate without changing its threshold.
8. Replace the final TODO only from committed DTU artifacts, then run a second
   paper-only reviewer red-team against the three ICLR writing benchmarks and
   the honest 6--8 completion bar.

## Handoff prompt for the vircs Codex session

> Work autonomously on `/root/Desktop/lyb/camcanon3r`.  Read
> `docs/VIRCS_HANDOFF.md`, `docs/RESEARCH_CONTRACT.md`, and
> `docs/EXPERIMENT_RUNBOOK.md` first.  This machine is the control plane only;
> execute GPU work on `my5090` through `scripts/vircs_remote_my5090.sh`.
> Preserve all existing code, Claude/Codex history, datasets, weights, outputs,
> and active jobs.  Before launching anything, verify both Git worktrees, GPU
> ownership, and the current ETH3D downloader state.  Keep code continuously
> committed and pushed to GitHub, bring only lightweight evidence into Git,
> and never use a global proxy or TUN.  Continue until the 3DV submission has
> hard multi-scene, multi-model, ground-truth evidence and an honest reviewer
> red-team supports an ICLR-style 6--8 score; do not call the goal complete
> based on scaffolding or a draft alone.
