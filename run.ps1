$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Get-PythonExe {
    $candidates = @(
        (Join-Path $root ".venv39\Scripts\python.exe"),
        (Join-Path $root ".venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return "python"
    }
    throw "Python bulunamadi. .venv39/.venv olustur veya PATH'e python ekle."
}

function Stop-Ports {
    param([int[]]$Ports)

    $pids = New-Object System.Collections.Generic.HashSet[int]
    foreach ($port in $Ports) {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $connections) {
            if ($conn.OwningProcess -gt 0) {
                [void]$pids.Add([int]$conn.OwningProcess)
            }
        }
    }

    foreach ($procId in $pids) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "Port process kapatildi: PID $procId"
        } catch {
            Write-Host "PID $procId kapatilamadi: $($_.Exception.Message)"
        }
    }
}

$python = Get-PythonExe

$npm = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npm) {
    throw "npm bulunamadi. Node.js kurulu olmali."
}

# Eski servisleri kapat
Stop-Ports -Ports @(8000, 5173, 5174, 5175, 8501)
Start-Sleep -Seconds 1

$rootEsc = $root -replace "'", "''"
$frontendPath = Join-Path $root "frontend"
$frontendEsc = $frontendPath -replace "'", "''"
$pythonEsc = $python -replace "'", "''"

# Backend
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$rootEsc'; & '$pythonEsc' -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload"
)

Start-Sleep -Seconds 2

# Frontend
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$frontendEsc'; npm run dev -- --host 127.0.0.1 --port 5173 --strictPort"
)

Write-Host ""
Write-Host "Backend : http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host ""
Write-Host "Calistirmak icin: .\run.ps1"
