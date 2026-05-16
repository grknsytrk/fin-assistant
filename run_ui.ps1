$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = ".\.external\venv311\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python ortami bulunamadi: $python"
    Write-Output "Kurulum ornegi: py -3.11 -m venv .external\venv311"
    exit 1
}

$env:VIRTUAL_ENV = (Resolve-Path ".\.external\venv311").Path
$env:PATH = "$env:VIRTUAL_ENV\Scripts;$env:PATH"
$env:PYTHONPATH = "$env:VIRTUAL_ENV\Lib\site-packages;$root"

& $python -m streamlit run app/ui.py
