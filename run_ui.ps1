param(
    [switch]$SkipInstall,
    [switch]$Reload,
    [switch]$NoLogWindows
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Backward-compatible launcher: the active application is React + FastAPI.
$runScript = Join-Path $root "run.ps1"
if (-not (Test-Path -LiteralPath $runScript)) {
    Write-Error "Aktif gelistirme launcher'i bulunamadi: $runScript"
    exit 1
}

& $runScript @PSBoundParameters
exit $LASTEXITCODE
