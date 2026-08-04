#!/usr/bin/env bash
set -euo pipefail

# Register and start a triggerless Windows Scheduled Task whose wsl.exe process
# owns one long-running Ubuntu command. A Linux-only tmux/nohup process is not
# sufficient on this host because WSL may stop when the last Windows client
# exits.

usage() {
  echo "usage: $0 CamCanon3R-JOB-NAME 'BASH COMMAND'" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

task_name="$1"
command_text="$2"
template_task="${CAMCANON3R_WINDOWS_TASK_TEMPLATE:-CamCanon3R-WSLDownloadProxy}"

if [[ ! "${task_name}" =~ ^CamCanon3R-[A-Za-z0-9_-]+$ ]]; then
  echo "task name must match CamCanon3R-[A-Za-z0-9_-]+" >&2
  exit 2
fi
if ! command -v powershell.exe >/dev/null 2>&1; then
  echo "powershell.exe is unavailable; this launcher must run inside my5090 WSL" >&2
  exit 1
fi

command_payload="$({ printf '%s' "${command_text}" | base64; } | tr -d '\n')"
argument_line="-d Ubuntu -- bash -lc \"printf %s ${command_payload} | base64 -d | bash\""

powershell_code="$(printf '%s\n' \
  '$ErrorActionPreference = "Stop"' \
  '$ProgressPreference = "SilentlyContinue"' \
  "\$taskName = '${task_name}'" \
  "\$templateName = '${template_task}'" \
  "\$desiredArguments = '${argument_line}'" \
  '$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue' \
  'if ($existing) {' \
  '  if ($existing.State -eq "Running") { throw "Task is already running: $taskName" }' \
  '  $existingAction = @($existing.Actions)[0]' \
  '  if ($existingAction.Execute -ne "wsl.exe" -or $existingAction.Arguments -ne $desiredArguments) {' \
  '    throw "Task exists with a different command: $taskName"' \
  '  }' \
  '} else {' \
  '  $template = Get-ScheduledTask -TaskName $templateName -ErrorAction Stop' \
  '  $action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument $desiredArguments' \
  '  Register-ScheduledTask -TaskName $taskName -Action $action -Principal $template.Principal -Settings $template.Settings -Description "Detached CamCanon3R WSL job; no triggers." | Out-Null' \
  '}' \
  'Start-ScheduledTask -TaskName $taskName' \
  'Start-Sleep -Seconds 2' \
  'Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State' \
)"

encoded_powershell="$({ printf '%s' "${powershell_code}" | iconv -f UTF-8 -t UTF-16LE | base64; } | tr -d '\n')"
powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand \
  "${encoded_powershell}"
