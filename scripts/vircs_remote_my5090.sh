#!/usr/bin/env bash
set -euo pipefail

# Run one shell command inside the Ubuntu WSL distribution on my5090.
#
# The Windows OpenSSH server inserts a cmd.exe hop.  Passing an arbitrary
# command directly through that hop is fragile because cmd.exe consumes shell
# operators before bash sees them.  Encoding the command and decoding it in a
# tiny Python launcher inside WSL keeps quoting deterministic.

host="${CAMCANON3R_MY5090_HOST:-my5090}"
timeout_seconds="${CAMCANON3R_SSH_TIMEOUT_SECONDS:-120}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 'shell command to run inside my5090 WSL'" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  command_text="$1"
else
  printf -v command_text '%q ' "$@"
fi

payload="$({ printf '%s' "$command_text" | base64; } | tr -d '\n')"
python_code="import base64,subprocess,sys; command=base64.b64decode('${payload}').decode('utf-8'); sys.exit(subprocess.run(['/bin/bash','-lc',command]).returncode)"

exec timeout "$timeout_seconds" ssh \
  -o BatchMode=yes \
  -o ConnectTimeout=15 \
  -o ServerAliveInterval=20 \
  -o ServerAliveCountMax=6 \
  "$host" \
  "wsl.exe -d Ubuntu -- python3 -c \"${python_code}\""
