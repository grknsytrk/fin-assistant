param(
    [switch]$SkipInstall,
    [switch]$Reload,
    [switch]$NoLogWindows
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvDir = Join-Path $root ".external\venv311"
$python = Join-Path $venvDir "Scripts\python.exe"
$frontendPath = Join-Path $root "frontend"
$logDir = Join-Path $root "logs"
$backendOut = Join-Path $logDir "backend.out.log"
$backendErr = Join-Path $logDir "backend.err.log"
$frontendOut = Join-Path $logDir "frontend.out.log"
$frontendErr = Join-Path $logDir "frontend.err.log"
$backendLog = Join-Path $logDir "backend.log"
$frontendLog = Join-Path $logDir "frontend.log"

function Write-Step {
    param([string]$Message)
    Write-Host "[rag-fin] $Message"
}

function ConvertTo-PsLiteral {
    param([string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Restore-ProcessEnv {
    param([hashtable]$Snapshot)

    foreach ($key in $Snapshot.Keys) {
        if ($null -eq $Snapshot[$key]) {
            Remove-Item -Path "Env:$key" -ErrorAction SilentlyContinue
        } else {
            Set-Item -Path "Env:$key" -Value $Snapshot[$key]
        }
    }
}

function Test-PythonRuntime {
    param([string]$PythonExe)
    if (-not (Test-Path $PythonExe)) {
        return $false
    }
    & $PythonExe -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-PythonVenv {
    if (-not (Test-Path $python)) {
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $pyLauncher) {
            throw "Python launcher bulunamadi. Python 3.11 kur ve tekrar .\run.ps1 calistir."
        }

        Write-Step "Python 3.11 venv olusturuluyor: $venvDir"
        $parent = Split-Path -Parent $venvDir
        if (-not (Test-Path $parent)) {
            New-Item -ItemType Directory -Path $parent | Out-Null
        }
        & py -3.11 -m venv $venvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 venv olusturulamadi. `py -3.11` komutunun calistigindan emin ol."
        }
    }

    if (-not (Test-PythonRuntime $python)) {
        throw "Backend Python 3.10-3.12 ister. Bu proje icin $python gecersiz. .external\venv311 klasorunu silip .\run.ps1 calistir."
    }
}

function Ensure-PythonDependencies {
    if ($SkipInstall) {
        Write-Step "Python dependency kontrolu atlandi."
        return
    }

    $requirements = Join-Path $root "requirements.txt"
    $stamp = Join-Path $venvDir ".ragfin_requirements.stamp"
    $needsInstall = -not (Test-Path $stamp)

    if ((Test-Path $requirements) -and (Test-Path $stamp)) {
        $needsInstall = (Get-Item $stamp).LastWriteTimeUtc -lt (Get-Item $requirements).LastWriteTimeUtc
    }

    & $python -c "import fastapi, uvicorn, tefasfon" *> $null
    if ($LASTEXITCODE -ne 0) {
        $needsInstall = $true
    }

    if ($needsInstall) {
        Write-Step "Python paketleri yukleniyor/guncelleniyor..."
        & $python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) {
            throw "pip guncellenemedi."
        }
        & $python -m pip install -r $requirements
        if ($LASTEXITCODE -ne 0) {
            throw "requirements.txt yuklenemedi."
        }
        Set-Content -Path $stamp -Value (Get-Date).ToUniversalTime().ToString("o") -Encoding ASCII
    }
}

function Ensure-FrontendDependencies {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) {
        $npm = Get-Command npm -ErrorAction SilentlyContinue
    }
    if (-not $npm) {
        throw "npm bulunamadi. Node.js kurulu olmali."
    }

    if ($SkipInstall) {
        Write-Step "Frontend dependency kontrolu atlandi."
        return $npm.Source
    }

    $nodeModules = Join-Path $frontendPath "node_modules"
    if (-not (Test-Path $nodeModules)) {
        Write-Step "Frontend paketleri yukleniyor..."
        & $npm.Source install --prefix $frontendPath
        if ($LASTEXITCODE -ne 0) {
            throw "frontend npm install basarisiz."
        }
    }

    return $npm.Source
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
            Write-Step "Port process kapatildi: PID $procId"
        } catch {
            Write-Step "PID $procId kapatilamadi: $($_.Exception.Message)"
        }
    }
}

function Stop-RagFinProcesses {
    $rootNeedle = $root.ToLowerInvariant()
    $targets = Get-CimInstance Win32_Process | Where-Object {
        if (-not $_.CommandLine) {
            return $false
        }
        if ($_.ProcessId -eq $PID) {
            return $false
        }
        $cmd = $_.CommandLine.ToLowerInvariant()
        return (
            $cmd.Contains("uvicorn app.api:app") -or
            ($cmd.Contains("\frontend") -and $cmd.Contains("npm run dev")) -or
            $cmd.Contains("rag-fin backend log window") -or
            $cmd.Contains("rag-fin frontend log window") -or
            ($cmd.Contains($rootNeedle) -and $cmd.Contains("vite") -and $cmd.Contains("--host 127.0.0.1"))
        )
    }

    foreach ($proc in $targets) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            Write-Step "Eski rag-fin process kapatildi: PID $($proc.ProcessId)"
        } catch {
            Write-Step "PID $($proc.ProcessId) kapatilamadi: $($_.Exception.Message)"
        }
    }

    $orphans = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
        if (-not $_.CommandLine) {
            return $false
        }
        $cmd = $_.CommandLine.ToLowerInvariant()
        if (-not $cmd.Contains("spawn_main(parent_pid=")) {
            return $false
        }
        if ($cmd -notmatch "spawn_main\(parent_pid=(\d+)") {
            return $false
        }
        $parentPid = [int]$matches[1]
        return -not (Get-Process -Id $parentPid -ErrorAction SilentlyContinue)
    }

    foreach ($proc in $orphans) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            Write-Step "Eski orphan Python worker kapatildi: PID $($proc.ProcessId)"
        } catch {
            Write-Step "PID $($proc.ProcessId) kapatilamadi: $($_.Exception.Message)"
        }
    }

    Stop-Ports -Ports @(8000, 5173, 5174, 5175, 8501)
    Start-Sleep -Seconds 1
}

function Wait-Http {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

function Start-Backend {
    if ($NoLogWindows) {
        Start-Process -FilePath $python `
            -ArgumentList $backendArgs `
            -WorkingDirectory $root `
            -WindowStyle Hidden `
            -RedirectStandardOutput $backendOut `
            -RedirectStandardError $backendErr
        return
    }

    Set-Content -Path $backendLog -Value "" -Encoding Unicode
    $rootLiteral = ConvertTo-PsLiteral $root
    $pythonLiteral = ConvertTo-PsLiteral $python
    $backendLogLiteral = ConvertTo-PsLiteral $backendLog
    $reloadFlag = if ($Reload) { " --reload" } else { "" }
    $command = @"
`$marker = 'rag-fin backend log window'
`$Host.UI.RawUI.WindowTitle = 'rag-fin backend'
Set-Location -LiteralPath $rootLiteral
Write-Host '[rag-fin] Backend loglari'
Write-Host '[rag-fin] http://127.0.0.1:8000'
& $pythonLiteral -m uvicorn app.api:app --host 127.0.0.1 --port 8000$reloadFlag 2>&1 | ForEach-Object { `$_.ToString() } | Tee-Object -FilePath $backendLogLiteral -Append
"@

    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $command) `
        -WorkingDirectory $root
}

function Start-Frontend {
    if ($NoLogWindows) {
        Start-Process -FilePath $npmExe `
            -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173", "--strictPort") `
            -WorkingDirectory $frontendPath `
            -WindowStyle Hidden `
            -RedirectStandardOutput $frontendOut `
            -RedirectStandardError $frontendErr
        return
    }

    Set-Content -Path $frontendLog -Value "" -Encoding Unicode
    $frontendLiteral = ConvertTo-PsLiteral $frontendPath
    $npmLiteral = ConvertTo-PsLiteral $npmExe
    $frontendLogLiteral = ConvertTo-PsLiteral $frontendLog
    $command = @"
`$marker = 'rag-fin frontend log window'
`$Host.UI.RawUI.WindowTitle = 'rag-fin frontend'
Set-Location -LiteralPath $frontendLiteral
Write-Host '[rag-fin] Frontend loglari'
Write-Host '[rag-fin] http://127.0.0.1:5173'
& $npmLiteral run dev -- --host 127.0.0.1 --port 5173 --strictPort 2>&1 | ForEach-Object { `$_.ToString() } | Tee-Object -FilePath $frontendLogLiteral -Append
"@

    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $command) `
        -WorkingDirectory $frontendPath
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Ensure-PythonVenv
Ensure-PythonDependencies
$npmExe = Ensure-FrontendDependencies

$envSnapshot = @{
    VIRTUAL_ENV = $env:VIRTUAL_ENV
    PATH = $env:PATH
    PYTHONPATH = $env:PYTHONPATH
}

try {
    $env:VIRTUAL_ENV = $venvDir
    $env:PATH = "$venvDir\Scripts;$env:PATH"
    $env:PYTHONPATH = "$venvDir\Lib\site-packages;$root"

    Write-Step "Eski servisler temizleniyor..."
    Stop-RagFinProcesses

    $backendArgs = @("-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000")
    if ($Reload) {
        $backendArgs += "--reload"
    }

    Write-Step "Backend baslatiliyor..."
    Start-Backend

    Write-Step "Frontend baslatiliyor..."
    Start-Frontend

    $backendReady = Wait-Http -Url "http://127.0.0.1:8000/health" -TimeoutSeconds 45
    $frontendReady = Wait-Http -Url "http://127.0.0.1:5173" -TimeoutSeconds 45
} finally {
    Restore-ProcessEnv $envSnapshot
}

Write-Host ""
if ($backendReady) {
    Write-Host "Backend : http://127.0.0.1:8000"
} else {
    $backendProblemLog = if ($NoLogWindows) { $backendErr } else { $backendLog }
    Write-Host "Backend : baslatildi ama health check gecmedi. Log: $backendProblemLog"
}
if ($frontendReady) {
    Write-Host "Frontend: http://127.0.0.1:5173"
} else {
    $frontendProblemLog = if ($NoLogWindows) { $frontendErr } else { $frontendLog }
    Write-Host "Frontend: baslatildi ama hazir gorunmuyor. Log: $frontendProblemLog"
}
if ($NoLogWindows) {
    Write-Host "Loglar   : $logDir"
} else {
    Write-Host "Loglar   : $backendLog"
    Write-Host "           $frontendLog"
}
Write-Host ""
