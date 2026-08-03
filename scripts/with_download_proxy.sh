#!/usr/bin/env bash
set -uo pipefail

# Run one command through an on-demand download proxy. On my5090 WSL this
# starts a triggerless Windows task bound only to the WSL virtual adapter. A
# local Mihomo process remains available as a portable fallback.

backend="${CAMCANON3R_DOWNLOAD_PROXY_BACKEND:-auto}"
proxy_home="${CAMCANON3R_DOWNLOAD_PROXY_HOME:-/opt/camcanon3r/.private-download-proxy}"
proxy_bin="${CAMCANON3R_DOWNLOAD_PROXY_BIN:-${proxy_home}/mihomo}"
proxy_config="${CAMCANON3R_DOWNLOAD_PROXY_CONFIG:-${proxy_home}/config.yaml}"
windows_task="${CAMCANON3R_DOWNLOAD_PROXY_TASK:-CamCanon3R-WSLDownloadProxy}"
proxy_log="${proxy_home}/mihomo.log"
proxy_pid=""
started_backend=""

usage() {
  echo "Usage: $0 COMMAND [ARG ...]" >&2
  echo "Runs COMMAND with a temporary, process-scoped download proxy." >&2
}

windows_task_exists() {
  command -v powershell.exe >/dev/null 2>&1 &&
    powershell.exe -NoLogo -NoProfile -NonInteractive -Command \
      "if (Get-ScheduledTask -TaskName '${windows_task}' -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" \
      >/dev/null 2>&1
}

cleanup() {
  case "${started_backend}" in
    windows)
      powershell.exe -NoLogo -NoProfile -NonInteractive -Command \
        "Stop-ScheduledTask -TaskName '${windows_task}' -ErrorAction SilentlyContinue; \
        Start-Sleep -Milliseconds 200; \
        \$listenerPids=@(Get-NetTCPConnection -LocalPort ${proxy_port} -State Listen -ErrorAction SilentlyContinue | \
        Select-Object -ExpandProperty OwningProcess -Unique); \
        foreach (\$processId in \$listenerPids) { \
          Stop-Process -Id \$processId -Force -ErrorAction SilentlyContinue \
        }" \
        >/dev/null 2>&1 || true
      ;;
    local)
      if [[ -n "${proxy_pid}" ]] && kill -0 "${proxy_pid}" 2>/dev/null; then
        kill "${proxy_pid}" 2>/dev/null || true
        wait "${proxy_pid}" 2>/dev/null || true
      fi
      ;;
  esac
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

if [[ "${backend}" == "auto" ]]; then
  if windows_task_exists; then
    backend="windows"
  else
    backend="local"
  fi
fi

case "${backend}" in
  windows)
    if ! windows_task_exists; then
      echo "Windows download-proxy task is unavailable: ${windows_task}" >&2
      exit 1
    fi
    proxy_host="$(ip route show default | awk '{print $3; exit}')"
    proxy_port="${CAMCANON3R_DOWNLOAD_PROXY_PORT:-17893}"
    if [[ -z "${proxy_host}" ]]; then
      echo "Could not resolve the Windows host address from WSL." >&2
      exit 1
    fi
    ;;
  local)
    proxy_host="127.0.0.1"
    proxy_port="${CAMCANON3R_DOWNLOAD_PROXY_PORT:-17892}"
    if [[ ! -x "${proxy_bin}" ]]; then
      echo "Download proxy binary is missing or not executable: ${proxy_bin}" >&2
      exit 1
    fi
    if [[ ! -r "${proxy_config}" ]]; then
      echo "Download proxy config is missing or unreadable: ${proxy_config}" >&2
      exit 1
    fi
    if awk '
      /^tun:[[:space:]]*$/ { in_tun = 1; next }
      in_tun && /^[^[:space:]]/ { in_tun = 0 }
      in_tun && /^[[:space:]]+enable:[[:space:]]*true([[:space:]]|$)/ { enabled = 1 }
      END { exit enabled ? 0 : 1 }
    ' "${proxy_config}"; then
      echo "Refusing to start: the private config enables TUN." >&2
      exit 1
    fi
    ;;
  *)
    echo "Unknown download-proxy backend: ${backend}" >&2
    exit 2
    ;;
esac

exec 9>"/tmp/camcanon3r-download-proxy.lock"
if ! flock -n 9; then
  echo "Another CamCanon3R download-proxy command is already running." >&2
  exit 1
fi

trap cleanup EXIT INT TERM

if [[ "${backend}" == "windows" ]]; then
  powershell.exe -NoLogo -NoProfile -NonInteractive -Command \
    "\$ErrorActionPreference='Stop'; Start-ScheduledTask -TaskName '${windows_task}'" \
    >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    echo "Failed to start Windows download-proxy task: ${windows_task}" >&2
    exit 1
  fi
  started_backend="windows"
else
  mkdir -p "${proxy_home}"
  chmod 700 "${proxy_home}"
  "${proxy_bin}" -d "${proxy_home}" -f "${proxy_config}" >"${proxy_log}" 2>&1 &
  proxy_pid=$!
  started_backend="local"
fi

ready=0
for _ in $(seq 1 50); do
  if [[ "${backend}" == "local" ]] && ! kill -0 "${proxy_pid}" 2>/dev/null; then
    echo "Download proxy exited before becoming ready; see ${proxy_log}" >&2
    exit 1
  fi
  if (echo >"/dev/tcp/${proxy_host}/${proxy_port}") >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "Download proxy did not listen on ${proxy_host}:${proxy_port}." >&2
  exit 1
fi

proxy_url="http://${proxy_host}:${proxy_port}"
HTTP_PROXY="${proxy_url}" \
HTTPS_PROXY="${proxy_url}" \
ALL_PROXY="${proxy_url}" \
http_proxy="${proxy_url}" \
https_proxy="${proxy_url}" \
all_proxy="${proxy_url}" \
NO_PROXY="localhost,127.0.0.1,::1" \
no_proxy="localhost,127.0.0.1,::1" \
  "$@"
command_status=$?

exit "${command_status}"
