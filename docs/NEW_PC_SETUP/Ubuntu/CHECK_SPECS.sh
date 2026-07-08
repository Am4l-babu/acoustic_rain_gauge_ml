#!/usr/bin/env bash
# ============================================================================
# CHECK NEW PC SPECS (Linux / Ubuntu) — run before the full
# master_feature_extraction.py run.
#
# Tells you a safe --workers value for this machine, checks GPU/CPU/RAM/disk,
# and confirms the HDD's contents survived the move.
#
# USAGE:
#   cd to wherever this HDD mounted, e.g. /media/$USER/E-HDD/acoustic_rain_gauge_ml
#   bash docs/NEW_PC_SETUP/Ubuntu/CHECK_SPECS.sh
#
# Automatically saves a copy of the report onto the HDD (alongside this
# script, as SPECS_REPORT_<hostname>_<date>.txt) while still printing it to
# the screen, via `main | tee ...` below - no separate command needed.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"      # .../acoustic_rain_gauge_ml
HDD_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"      # parent of acoustic_rain_gauge_ml (the HDD root)
REPORT_FILE="$SCRIPT_DIR/SPECS_REPORT_$(hostname)_$(date +%Y-%m-%d).txt"

main() {
echo "======================================================================"
echo "  0. OS"
echo "======================================================================"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "  ${PRETTY_NAME:-unknown}"
fi
uname -a
echo ""

echo "======================================================================"
echo "  1. RAM (decides safe --workers value)"
echo "======================================================================"
read -r _ total used free shared buffcache available < <(free -m | awk '/^Mem:/{print}')
echo "  Total RAM     : ${total} MB"
echo "  Available RAM : ${available} MB   (this is the number that matters — accounts for reclaimable cache)"

# Rule of thumb measured on the original (Windows) PC: ~0.5GB free RAM needed
# per worker (each worker imports numpy/scipy/librosa, ~200-400MB apiece).
# Linux's fork-based multiprocessing shares already-imported modules via
# copy-on-write rather than reimporting per worker like Windows' spawn method,
# so actual headroom per worker is usually better than this conservative
# estimate — treat this as a floor, not a hard ceiling.
safe_workers=$(( available / 512 ))
if [ "$safe_workers" -lt 1 ]; then safe_workers=1; fi
echo "  -> Suggested --workers (rule of thumb, ~0.5GB available RAM per worker): $safe_workers"
echo ""

echo "======================================================================"
echo "  2. CPU"
echo "======================================================================"
cores=$(nproc)
echo "  Logical cores: $cores"
echo "  Script's own cap (cpu_count()-2, before the RAM-based min(6, ...)): $(( cores - 2 ))"
if [ -f /proc/cpuinfo ]; then
    model=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')
    echo "  Model: $model"
fi
echo ""

echo "======================================================================"
echo "  3. GPU"
echo "======================================================================"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap \
        --format=csv,noheader 2>/dev/null | while IFS=, read -r name memtotal memfree driver cc; do
        echo "  NVIDIA GPU: $name"
        echo "    VRAM total: $memtotal   VRAM free: $memfree"
        echo "    Driver: $driver   Compute capability: $cc"
    done
elif command -v lspci >/dev/null 2>&1; then
    gpu=$(lspci | grep -iE 'vga|3d|display')
    if [ -n "$gpu" ]; then
        echo "  $gpu"
        echo "  (no nvidia-smi found — if this is an NVIDIA card, install the driver"
        echo "   before relying on GPU-accelerated training)"
    else
        echo "  No GPU detected via lspci."
    fi
else
    echo "  Neither nvidia-smi nor lspci available — install pciutils to check (sudo apt install pciutils)."
fi
echo ""

echo "======================================================================"
echo "  4. Disks + free space"
echo "======================================================================"
df -h --output=source,fstype,size,avail,pcent,target 2>/dev/null | grep -vE "^tmpfs|^udev|^overlay|^/dev/loop"
echo ""

echo "======================================================================"
echo "  5. HDD contents check"
echo "======================================================================"
echo "  HDD root (auto-detected from this script's own location): $HDD_ROOT"
for rel in "arg_dataset_unzip" "arg_cleaned_dataset" "acoustic_rain_gauge_ml/src/master_feature_extraction.py"; do
    path="$HDD_ROOT/$rel"
    if [ -e "$path" ]; then
        echo "  OK       $path"
    else
        echo "  MISSING  $path"
    fi
done
echo ""

echo "======================================================================"
echo "  SUMMARY"
echo "======================================================================"
echo "  Recommended command:"
echo "    cd $PROJECT_DIR"
echo "    python3 src/master_feature_extraction.py --limit 2000 --workers $safe_workers"
echo "  (run the smoke test first regardless - confirms real throughput on this machine"
echo "   before committing to the full ~7h run at this worker count)"
echo ""
echo "  (this report was saved to $REPORT_FILE)"
}

main | tee "$REPORT_FILE"
