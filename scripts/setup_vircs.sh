#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

available_kb="$(df -Pk . | awk 'NR == 2 {print $4}')"
if (( available_kb < 700000 )); then
  echo "ERROR: less than 700 MB is free; refusing to create an environment." >&2
  echo "Attach storage or obtain approval for a conservative cache cleanup." >&2
  exit 4
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q

echo "vircs CPU development environment is ready."
echo "GPU inference must be launched on my5090 through scripts/vircs_remote_my5090.sh."
