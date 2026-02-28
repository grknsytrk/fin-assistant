$python = ".\.venv39\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python ortami bulunamadi: $python"
    Write-Output "Kurulum ornegi: py -3.9 -m venv .venv39"
    exit 1
}

& $python -m streamlit run app/ui.py
