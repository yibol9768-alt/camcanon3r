#!/usr/bin/env bash
set -euo pipefail

repo="${CAMCANON3R_MY5090_REPO:-/opt/camcanon3r}"
branch="${CAMCANON3R_BRANCH:-main}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

read -r -d '' remote_command <<EOF || true
set -euo pipefail
cd '$repo'
if [[ -n "\$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'ERROR: tracked changes exist on my5090; refusing to overwrite them.' >&2
  git status --short
  exit 3
fi
if [[ -x ./scripts/with_download_proxy.sh ]]; then
  ./scripts/with_download_proxy.sh git fetch origin '$branch'
else
  git fetch origin '$branch'
fi
git checkout '$branch'
git merge --ff-only 'origin/$branch'
printf 'MY5090_HEAD='
git rev-parse HEAD
EOF

"$script_dir/vircs_remote_my5090.sh" "$remote_command"
