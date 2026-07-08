#!/usr/bin/env bash
# ============================================================================
# RUN MASTER FEATURE EXTRACTION (Linux / Ubuntu) — automated: install deps,
# smoke test, validate, full run.
# ============================================================================
# Run this FROM the project folder on the HDD (mount point auto-detected from
# wherever this script lives — no need to hardcode a path).
#
# USAGE:
#   cd /media/$USER/E-HDD/acoustic_rain_gauge_ml   (or wherever the HDD mounted)
#   bash docs/NEW_PC_SETUP/Ubuntu/RUN_EXTRACTION.sh
#
# What it does:
#   1. pip install -r requirements.txt
#   2. Smoke test: --limit 2000
#   3. Validates the smoke test actually produced non-empty output (guards
#      against the silent "0 files found" failure mode if the HDD mount path
#      is wrong)
#   4. If the smoke test passed, runs the full extraction automatically
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "======================================================================"
echo "  MASTER FEATURE EXTRACTION — automated run"
echo "  Project root: $PROJECT_DIR"
echo "======================================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found — install it first (sudo apt install python3 python3-pip)."
    exit 1
fi

echo ""
echo "Step 1/3: Installing dependencies..."
python3 -m pip install -q -r "$PROJECT_DIR/requirements.txt"
if [ $? -ne 0 ]; then
    echo "pip install failed - aborting."
    exit 1
fi
echo "  Dependencies installed."

echo ""
echo "Step 2/3: Smoke test (2000 clips)..."
python3 "$PROJECT_DIR/src/master_feature_extraction.py" --limit 2000
if [ $? -ne 0 ]; then
    echo "Smoke test crashed - aborting full run. Check the error above."
    exit 1
fi

# Guard against the silent "0 valid files" failure mode (wrong mount path, HDD
# not connected, etc.) — the script itself warns loudly, but double-check here
# before committing to the multi-hour full run. Asks the script itself for its
# OUTPUT_DIR rather than recomputing the path logic separately, so this can't
# drift out of sync with master_feature_extraction.py's actual behavior.
check_result=$(cd "$PROJECT_DIR" && python3 -c "
import sys
sys.path.insert(0, 'src')
from master_feature_extraction import OUTPUT_DIR
import pandas as pd
files = sorted(OUTPUT_DIR.glob('master_chunk_*.parquet'))
if not files:
    print('NONE')
    sys.exit(1)
n = sum(len(pd.read_parquet(f)) for f in files)
print(n)
sys.exit(0 if n > 0 else 1)
")
check_status=$?
if [ $check_status -ne 0 ] || [ "$check_result" = "NONE" ] || [ "${check_result:-0}" -eq 0 ] 2>/dev/null; then
    echo "Smoke test produced no usable rows - aborting full run."
    echo "Check that the HDD is mounted and its contents are where the project expects (run CHECK_SPECS.sh)."
    exit 1
fi
echo "  Smoke test OK ($check_result rows) - proceeding to full run."

echo ""
echo "Step 3/3: Full extraction (all clips, ETA depends on --workers - see CHECK_SPECS.sh)..."
python3 "$PROJECT_DIR/src/master_feature_extraction.py"
if [ $? -ne 0 ]; then
    echo "Full extraction failed - see error above."
    exit 1
fi

echo ""
echo "======================================================================"
echo "  DONE - master feature store complete."
echo "======================================================================"
