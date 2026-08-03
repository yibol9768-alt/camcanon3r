$ErrorActionPreference = 'Stop'

$proxyDirectory = Join-Path $env:LOCALAPPDATA 'CamCanon3RDownloadProxy'
$baseConfig = Join-Path $proxyDirectory 'config.yaml'
$runtimeConfig = Join-Path $proxyDirectory 'config-wsl.yaml'
$proxyExecutable = Join-Path $env:USERPROFILE 'camcanon3r-download-proxy\mihomo.exe'
$proxyPort = 17893

$wslAddress = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -like '*WSL*' } |
    Select-Object -First 1 -ExpandProperty IPAddress

if (-not $wslAddress) {
    throw 'Could not find the WSL virtual-adapter IPv4 address.'
}

$configText = Get-Content -Raw $baseConfig
$tunBlock = [regex]::Match(
    $configText,
    '(?ms)^tun:\s*\r?\n(?<body>(?:[ \t]+.*\r?\n?)*)'
)
if ($tunBlock.Success -and $tunBlock.Groups['body'].Value -match '(?m)^\s+enable:\s*true\s*$') {
    throw 'Refusing to start because the private configuration enables TUN.'
}

$configText = [regex]::Replace(
    $configText,
    '(?m)^mixed-port:\s*\d+\s*$',
    "mixed-port: $proxyPort"
)
$configText = [regex]::Replace(
    $configText,
    '(?m)^allow-lan:\s*\S+\s*$',
    'allow-lan: true'
)
$configText = [regex]::Replace(
    $configText,
    '(?m)^bind-address:\s*\S+\s*$',
    "bind-address: $wslAddress"
)

$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($runtimeConfig, $configText, $utf8WithoutBom)

& $proxyExecutable -d $proxyDirectory -f $runtimeConfig
exit $LASTEXITCODE
