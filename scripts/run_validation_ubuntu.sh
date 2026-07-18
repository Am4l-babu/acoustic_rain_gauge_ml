#!/usr/bin/env bash
# Runs the Stage 9 held-out validation (src/evaluation/validate_new_data.py)
# unattended on Ubuntu, against whatever drive this script's own
# acoustic_rain_gauge_ml folder lives on. Designed to be copied onto the HDD
# itself (F:\acoustic_rain_gauge_ml\scripts\ on Windows) so it travels with
# the data and works regardless of which mount point the drive gets on the
# Ubuntu machine that plugs it in.
#
# Usage (once the HDD is mounted on Ubuntu, e.g. /media/$USER/E-HDD):
#   cd /media/$USER/E-HDD/acoustic_rain_gauge_ml
#   bash scripts/run_validation_ubuntu.sh
#
# Everything -- the venv, logs, chunk checkpoints, final report and plots --
# is written under this same acoustic_rain_gauge_ml folder on the HDD, so
# nothing touches the host machine's disk except pip's package cache.
#
# Safe to re-run: validate_new_data.py skips chunks it already finished
# (docs/reports/new_data_validation_chunks/chunk_*.parquet), so an
# interrupted run resumes instead of restarting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"          # .../acoustic_rain_gauge_ml
HDD_ROOT="$(dirname "$REPO_ROOT")"            # sibling of arg_dataset_unzip
VENV_DIR="$REPO_ROOT/.venv_linux"
LOG_DIR="$REPO_ROOT/docs/reports"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/validation_run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "=================================================================="
echo " Stage 9 validation — Ubuntu unattended runner"
echo " Repo root : $REPO_ROOT"
echo " HDD root  : $HDD_ROOT  (expects arg_dataset_unzip/ here)"
echo " Log file  : $LOG_FILE"
echo "=================================================================="

if [ ! -d "$HDD_ROOT/arg_dataset_unzip" ]; then
    echo "WARNING: $HDD_ROOT/arg_dataset_unzip not found." \
         "Pass --audio-root explicitly if the dataset lives elsewhere on this machine." | tee -a "$LOG_FILE"
fi

# ---- venv setup (created once, reused on subsequent runs) ----
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating Linux venv at $VENV_DIR ..." | tee -a "$LOG_FILE"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[setup] Installing/checking dependencies..." | tee -a "$LOG_FILE"
pip install --quiet --upgrade pip
# Everything except torch/torchaudio (CUDA build varies by machine, installed next)
grep -v -E '^torch(audio)?==' "$REPO_ROOT/requirements.txt" | pip install --quiet -r /dev/stdin

# CUDA 12.4 build to match this project's training environment. If this
# machine has a different CUDA driver, adjust the index URL/version here
# (check with `nvidia-smi`) or drop --index-url entirely for the CPU build.
if ! python3 -c "import torch" 2>/dev/null; then
    echo "[setup] Installing torch/torchaudio (cu124 build)..." | tee -a "$LOG_FILE"
    pip install --quiet torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchinfo==1.8.0 \
        --index-url https://download.pytorch.org/whl/cu124
fi

python3 -c "import torch; print('[setup] torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())" | tee -a "$LOG_FILE"

# ---- run the validation, unattended, logging to the HDD ----
echo "[run] Launching validate_new_data.py ..." | tee -a "$LOG_FILE"
cd "$REPO_ROOT"
nohup python3 src/evaluation/validate_new_data.py "$@" >> "$LOG_FILE" 2>&1 &
RUN_PID=$!
disown

echo "=================================================================="
echo " Started in background, PID $RUN_PID (survives terminal close)."
echo " Tail progress with:"
echo "   tail -f \"$LOG_FILE\""
echo " Results land in:"
echo "   $REPO_ROOT/docs/reports/new_data_validation_report.json"
echo "   $REPO_ROOT/docs/reports/new_data_validation_*.png"
echo "=================================================================="
