# ============================================================================
# RUN MASTER FEATURE EXTRACTION — automated: install deps, smoke test, full run
# ============================================================================
# Run this FROM the project folder on the HDD (drive letter auto-detected from
# wherever this script lives — no need to hardcode F: or reassign drive letters).
#
# USAGE:
#   cd F:\acoustic_rain_gauge_ml   (or wherever the HDD mounted)
#   .\docs\NEW_PC_SETUP\Windows\RUN_EXTRACTION.ps1
#
# What it does:
#   1. pip install -r requirements.txt
#   2. Smoke test: --limit 2000
#   3. Validates the smoke test actually produced non-empty output (guards
#      against the silent "0 files found" failure mode if the HDD path/drive
#      letter is wrong)
#   4. If the smoke test passed, runs the full extraction automatically
# ============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Item "$PSScriptRoot\..\..\..").FullName

Write-Host "======================================================================"
Write-Host "  MASTER FEATURE EXTRACTION — automated run"
Write-Host "  Project root: $ProjectRoot"
Write-Host "======================================================================"

Write-Host "`nStep 1/3: Installing dependencies..."
python -m pip install -q -r "$ProjectRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip install failed — aborting."
    exit 1
}
Write-Host "  Dependencies installed."

Write-Host "`nStep 2/3: Smoke test (2000 clips)..."
python "$ProjectRoot\src\master_feature_extraction.py" --limit 2000
if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke test crashed — aborting full run. Check the error above."
    exit 1
}

# Guard against the silent "0 valid files" failure mode (wrong drive letter,
# HDD not connected, etc.) — the script itself warns loudly, but double-check
# here before committing to the multi-hour full run.
$checkScript = @"
import glob, sys
from pathlib import Path
self_drive = Path(r'$ProjectRoot').drive
files = glob.glob(str(Path(self_drive) / 'master_feature_store' / 'master_chunk_*.parquet'))
if not files:
    print('No output chunks found.')
    sys.exit(1)
import pandas as pd
n = sum(len(pd.read_parquet(f)) for f in files)
print(f'{n} rows found in {len(files)} chunk(s).')
sys.exit(0 if n > 0 else 1)
"@
$checkScript | python -
if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke test produced no usable rows — aborting full run."
    Write-Host "Check that the HDD is connected and --audio-root / --data-dir resolve correctly."
    exit 1
}
Write-Host "  Smoke test OK — proceeding to full run."

Write-Host "`nStep 3/3: Full extraction (all clips, ~1-2h)..."
python "$ProjectRoot\src\master_feature_extraction.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Full extraction failed — see error above."
    exit 1
}

Write-Host "`n======================================================================"
Write-Host "  DONE — master feature store complete."
Write-Host "======================================================================"
