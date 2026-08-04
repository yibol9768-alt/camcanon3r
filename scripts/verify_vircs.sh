#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
remote="$repo_root/scripts/vircs_remote_my5090.sh"
cd "$repo_root"

echo "== vircs control plane =="
printf 'host='
hostname
printf 'repo=%s\n' "$repo_root"
printf 'head='
git rev-parse HEAD
git status --short --branch
df -h .
python3 --version
git --version

if [[ -x .venv/bin/python ]]; then
  .venv/bin/python -m pytest -q
else
  echo "NOTE: .venv is absent; run ./scripts/setup_vircs.sh for CPU tests."
fi

echo
echo "== my5090 execution plane =="
read -r -d '' remote_command <<'EOF' || true
set -euo pipefail
printf 'host='
hostname
cd /opt/camcanon3r
printf 'repo=/opt/camcanon3r\nhead='
git rev-parse HEAD
git status --short --branch
printf 'gpu='
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PYTHONPATH=src .venv/bin/python -m pytest -q
printf '%s  %s\n' \
  '7c300a89534113436bde52732d3151212bcbd90f0aa3c8d1496f86d84bfe4b42' \
  'checkpoints/dust3r-512-dpt/model.safetensors' | sha256sum -c -
du -sh checkpoints/VGGT-1B checkpoints/dust3r-512-dpt data outputs results
df -h /opt /mnt/e
printf 'active_downloads='
pgrep -af '[d]ownload_archives.py' || true
EOF
"$remote" "$remote_command"

echo
echo "PASS: vircs can control the my5090 WSL experiment environment."
