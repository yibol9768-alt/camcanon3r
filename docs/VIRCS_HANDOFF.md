# CamCanon3R handoff: vircs controls, my5090 executes

Updated: 2026-08-04 (Asia/Shanghai)

## Mission and completion bar

Continue CamCanon3R as a 3DV paper project.  The work is not complete merely
because the code runs or the current draft compiles.  Stop only after the
paper, evidence, reproducibility package, and reviewer red-team jointly support
an estimated ICLR-style score in the 6--8 range.  Report uncertainty honestly;
never promote a claim beyond the committed evidence.

The present paper is an incomplete working draft.  Related Work and detector
results still contain TODOs.  The frozen three-scene diagnostic is evidence of
non-equivariance, not yet a ground-truth accuracy or multi-model result.

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

- Expected repository baseline at handoff creation:
  `a773e36f00049f862db919624ef84f9c3080ed5b` plus the handoff commit that
  contains this file.
- Local and my5090 CPU test baseline before this handoff: 47 tests passed.
- VGGT weights already reside on `my5090`.
- DUSt3R weights reside at
  `/opt/camcanon3r/checkpoints/dust3r-512-dpt`; the previously verified large
  weight SHA-256 is
  `7c300a89534113436bde52732d3151212bcbd90f0aa3c8d1496f86d84bfe4b42`.
- ETH3D office assets already reside on `my5090`.
- The manifest freezes all 13 official ETH3D training scenes.  Before starting
  another downloader, inspect the current process, archive directory, log, and
  `download_report.json`; do not create a duplicate job.
- Live verification on 2026-08-04 found no active `download_archives.py`
  process and an empty archive directory.  The old log ended with a missing
  local-Mihomo fallback error, but the preferred Windows scheduled-task backend
  subsequently completed a proxied Git fetch.  Recheck, then the frozen
  `./scripts/start_eth3d_download_my5090.sh` entry point may be started once.
- Live asset sizes at handoff were approximately 4.7 GB for VGGT weights,
  2.2 GB for DUSt3R weights, 85 MB for prepared data, and 212 MB for outputs.
  Both machines passed all 47 CPU tests when `my5090` used `PYTHONPATH=src`.

Re-verify every item live before reporting it as current.  Machine state and
download progress are allowed to drift after this document is committed.

## Safe experiment sequence

1. Run `./scripts/verify_vircs.sh` and record both commit hashes.
2. If hashes differ, inspect both worktrees; push from `vircs`, then use the
   fast-forward sync script.  Never overwrite uncommitted work.
3. On `my5090`, check `nvidia-smi` process list and utilization.  Do not start
   VGGT or DUSt3R while another project owns the GPU.
4. Check whether `download_archives.py` is already running before touching the
   ETH3D download task.
5. Follow `docs/EXPERIMENT_RUNBOOK.md` exactly. Use the Windows-owned background
   launcher for long commands, use resumable runners, and keep
   raw, transformed, repaired, and clean-control outputs distinct.
6. Bring only small machine-readable summaries, plots, provenance, and paper
   evidence back into Git.  Run the full CPU test suite before every push.
7. Update the paper only after claim gates in `docs/RESEARCH_CONTRACT.md`,
   `docs/PAPER_STORY.md`, and `docs/RELIABILITY_PROTOCOL.md` are satisfied.

## Immediate research queue

1. Inspect and, if needed, resume the frozen 13-scene ETH3D downloads.
2. Complete ground-truth ETH3D runs for the chosen transformations and paired
   scene-level bootstrap summaries.
3. Complete the DUSt3R confirmatory matrix without pooling it with VGGT.
4. Evaluate analytic repair against the same ground truth, including clean
   cost, visible support, negative recovery, and overshoot.
5. Populate reliability AUROC, risk--coverage, AURC/excess-AURC, and
   scene-cluster confidence intervals on held-out cases.
6. Replace paper TODOs only with committed results, then red-team the complete
   manuscript against 3DV scope and the requested ICLR-style 6--8 bar.

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
