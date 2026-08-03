#!/usr/bin/env bash
set -euo pipefail

repo_root="${CAMCANON3R_REPO_ROOT:-/opt/camcanon3r}"
archive_root="${CAMCANON3R_ETH3D_ARCHIVES:-/mnt/e/camcanon3r-data/eth3d_archives}"
manifest="${repo_root}/configs/eth3d_training_archives.json"
log_path="${archive_root}/download.log"

mkdir -p "${archive_root}"
exec >>"${log_path}" 2>&1
printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

exec 8>"${archive_root}/download.lock"
if ! flock -n 8; then
  echo "another ETH3D archive download holds ${archive_root}/download.lock"
  exit 1
fi

cd "${repo_root}"
exec ./scripts/with_download_proxy.sh \
  nice -n 15 ionice -c 3 \
  .venv/bin/python scripts/download_archives.py \
  "${manifest}" "${archive_root}" \
  --report "${archive_root}/download_report.json"
