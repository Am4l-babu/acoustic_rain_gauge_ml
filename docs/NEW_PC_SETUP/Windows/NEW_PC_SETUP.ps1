# ============================================================================
# NEW PC SETUP — FROM HDD
# ============================================================================
# Run this on the NEW PC after connecting the HDD
#
# USAGE (in PowerShell as Admin):
#   .\NEW_PC_SETUP.ps1 -HDD_Drive "E:" -LocalPath "D:\acoustic_rain_gauge_ml"
#
# This script will:
#   1. Copy project from HDD to local PC (fast operation)
#   2. Install Python dependencies (torch, etc.)
#   3. Verify GPU/CUDA
#   4. Ready for immediate training
# ============================================================================

param(
    [string]$HDD_Drive = "E:",
    [string]$LocalPath = "D:\acoustic_rain_gauge_ml",
    [bool]$SkipPythonInstall = $false
)

$HDD_ProjectPath = "$HDD_Drive\acoustic_rain_gauge_ml"

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║  NEW PC SETUP — ACOUSTIC RAIN GAUGE ML                        ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "📍 HDD location: $HDD_ProjectPath"
Write-Host "📍 Local installation: $LocalPath"
Write-Host ""

# ============================================================================
# STEP 1: Verify HDD is accessible
# ============================================================================

Write-Host "Step 1/5: Checking HDD connection..."

if (-not (Test-Path $HDD_ProjectPath)) {
    Write-Host "❌ ERROR: HDD not found at $HDD_ProjectPath"
    Write-Host "   Please:"
    Write-Host "   1. Connect the HDD to this PC"
    Write-Host "   2. Note the drive letter (e.g., E:, F:, etc.)"
    Write-Host "   3. Re-run: .\NEW_PC_SETUP.ps1 -HDD_Drive 'D:' (or your drive)"
    exit 1
}

Write-Host "  ✓ HDD found at $HDD_Drive"

# Verify essential files
$required_files = @("requirements.txt", "src/train_model.py", "src/train_dl_model.py")
foreach ($file in $required_files) {
    if (-not (Test-Path "$HDD_ProjectPath\$file")) {
        Write-Host "❌ ERROR: $file not found on HDD"
        Write-Host "   Did you run migrate_to_hdd.ps1 on the old PC?"
        exit 1
    }
}

Write-Host "  ✓ All essential files present on HDD"
Write-Host ""

# ============================================================================
# STEP 2: Copy project to local drive (faster for training)
# ============================================================================

Write-Host "Step 2/5: Copying project to local SSD (for faster training)..."

if (-not (Test-Path $LocalPath)) {
    New-Item -ItemType Directory -Path $LocalPath -Force | Out-Null
}

Write-Host "  ⏳ This may take 1-2 minutes..."
Copy-Item "$HDD_ProjectPath\*" "$LocalPath\" -Recurse -Force | Out-Null

Write-Host "  ✓ Project copied to: $LocalPath"
Write-Host ""

# ============================================================================
# STEP 3: Check Python installation
# ============================================================================

Write-Host "Step 3/5: Checking Python installation..."

$python_check = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found"
    Write-Host "   Please install Python 3.10+ from https://python.org/downloads"
    Write-Host "   IMPORTANT: Check 'Add Python to PATH' during installation"
    exit 1
}

Write-Host "  ✓ $python_check"
Write-Host ""

# ============================================================================
# STEP 4: Create virtual environment and install dependencies
# ============================================================================

Write-Host "Step 4/5: Setting up virtual environment..."

cd $LocalPath

# Create venv
if (-not (Test-Path ".venv")) {
    Write-Host "  ⏳ Creating virtual environment..."
    python -m venv .venv
    Write-Host "  ✓ Virtual environment created"
} else {
    Write-Host "  ✓ Virtual environment already exists"
}

# Activate venv
Write-Host "  ⏳ Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "  ✓ Virtual environment activated"
Write-Host ""

# Install base packages
Write-Host "Step 5/5: Installing Python dependencies..."
Write-Host "  ⏳ Installing standard ML stack (pip, numpy, pandas, xgboost, etc.)..."
pip install -q -r requirements.txt
Write-Host "  ✓ Standard packages installed"

# Install PyTorch (check for GPU)
Write-Host "  ⏳ Installing PyTorch (checking for GPU support)..."
$nvidia_check = nvidia-smi 2>$null
if ($nvidia_check) {
    Write-Host "  ✓ NVIDIA GPU detected! Installing PyTorch with CUDA support..."
    pip install -q torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
    Write-Host "  ✓ PyTorch (CUDA) installed"
} else {
    Write-Host "  ℹ No GPU detected. Installing PyTorch for CPU..."
    pip install -q torch torchaudio
    Write-Host "  ✓ PyTorch (CPU) installed"
}

Write-Host ""

# ============================================================================
# STEP 6: Verify GPU (if available)
# ============================================================================

Write-Host "🔍 Verifying GPU availability..."
python -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')" 2>$null

Write-Host ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║  ✅ SETUP COMPLETE! READY TO TRAIN                           ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

Write-Host "📂 Project location: $LocalPath"
Write-Host "📦 Dataset location: Update path in src/feature_extraction.py if needed"
Write-Host ""

Write-Host "🚀 Quick start commands:"
Write-Host ""

Write-Host "  1️⃣  Activate virtual environment (run in NEW PowerShell window):"
Write-Host "      cd $LocalPath"
Write-Host "      .\.venv\Scripts\Activate.ps1"
Write-Host ""

Write-Host "  2️⃣  Generate training data (if not already done, ~15-20 min):"
Write-Host "      python src/feature_extraction.py"
Write-Host ""

Write-Host "  3️⃣  Train XGBoost (fast, ~5 min):"
Write-Host "      python src/train_model.py"
Write-Host "      python src/evaluate_model.py"
Write-Host ""

Write-Host "  4️⃣  Train Transformer (best, ~1-2 hours with GPU):"
Write-Host "      python src/train_dl_model.py --model transformer --epochs 50 --batch_size 32"
Write-Host ""

Write-Host "  5️⃣  Check results:"
Write-Host "      cat docs\model_evaluation_report.json"
Write-Host ""

Write-Host "📍 Important paths:"
Write-Host "  • Dataset: Make sure D:\arg_cleaned_dataset\ exists (or on HDD)"
Write-Host "  • Data: $LocalPath\data\processed\"
Write-Host "  • Models: $LocalPath\models\"
Write-Host "  • Results: $LocalPath\docs\model_evaluation_report.json"
Write-Host ""

Write-Host "💡 If dataset is on the HDD:"
Write-Host "   Edit src/feature_extraction.py line 14:"
Write-Host "   DATA_DIR = Path(r'$HDD_Drive\arg_cleaned_dataset')"
Write-Host ""

Write-Host "✨ You're all set! Start training whenever you're ready."
Write-Host ""
