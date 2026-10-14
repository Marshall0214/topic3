param([string]$Python = 'C:\Users\admin\anaconda3\python.exe')
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NUMBA_CACHE_DIR = Join-Path $TaskRoot 'runs\_cache\numba'
$env:MPLCONFIGDIR = Join-Path $TaskRoot 'runs\_cache\matplotlib'
Push-Location -LiteralPath $TaskRoot
try {
    & $Python -B scripts/check_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment inventory failed' }
    & $Python -B -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Contract checks failed' }
    & $Python -B scripts/train.py --selfcheck
    if ($LASTEXITCODE -ne 0) { throw 'DeST synthetic selfcheck failed' }
    & $Python -B scripts/summarize_results.py
    if ($LASTEXITCODE -ne 0) { throw 'Saved-results summary failed' }
    & $Python -B scripts/verify_migration.py
    if ($LASTEXITCODE -ne 0) { throw 'Source/snapshot verification failed' }
} finally { Pop-Location }
