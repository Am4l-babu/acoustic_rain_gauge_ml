# ============================================================================
# MIGRATE ACOUSTIC RAIN GAUGE ML TO HDD
# ============================================================================
# This script copies all essential files (that aren't in GitHub) to an HDD
# so you can move to the new PC and start training immediately
#
# USAGE:
#   .\migrate_to_hdd.ps1 -TargetDrive "E:" -HDD_DataPath "E:\acoustic_rain_gauge_ml"
# ============================================================================

param(
    [string]$TargetDrive = "E:",
    [string]$HDD_DataPath = "E:\acoustic_rain_gauge_ml",
    [bool]$SkipDatasetCopy = $false  # Set to $true if dataset already on HDD
)

$REPO_ROOT = "D:\acoustic_rain_gauge_ml"

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║  ACOUSTIC RAIN GAUGE ML → HDD MIGRATION TOOL                 ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "📍 Source: $REPO_ROOT"
Write-Host "📍 Target: $HDD_DataPath"
Write-Host ""

# ============================================================================
# STEP 1: Verify HDD has enough space
# ============================================================================

Write-Host "Step 1/5: Checking HDD space..."
$drive_info = Get-Volume -DriveLetter $TargetDrive[0] -ErrorAction SilentlyContinue
if (-not $drive_info) {
    Write-Host "❌ Error: Drive $TargetDrive not found"
    exit 1
}

$free_space_gb = $drive_info.SizeRemaining / 1GB
Write-Host "  ✓ Free space on $TargetDrive : $([Math]::Round($free_space_gb, 1)) GB"

if ($free_space_gb -lt 5) {
    Write-Host "❌ Not enough space (need 5 GB minimum)"
    exit 1
}

# ============================================================================
# STEP 2: Create directory structure on HDD
# ============================================================================

Write-Host ""
Write-Host "Step 2/5: Creating directory structure on HDD..."

$dirs_to_create = @(
    "$HDD_DataPath",
    "$HDD_DataPath\src",
    "$HDD_DataPath\data",
    "$HDD_DataPath\data\processed",
    "$HDD_DataPath\models",
    "$HDD_DataPath\docs",
    "$HDD_DataPath\notebooks"
)

foreach ($dir in $dirs_to_create) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✓ Created: $dir"
    } else {
        Write-Host "  ✓ Exists: $dir"
    }
}

# ============================================================================
# STEP 3: Copy source code (from GitHub repo)
# ============================================================================

Write-Host ""
Write-Host "Step 3/5: Copying source code..."

$src_files = @(
    "requirements.txt",
    "README.md",
    ".gitignore",
    "setup.py"
)

foreach ($file in $src_files) {
    $src = Join-Path $REPO_ROOT $file
    $dst = Join-Path $HDD_DataPath $file
    if (Test-Path $src) {
        Copy-Item $src $dst -Force
        Write-Host "  ✓ $file"
    }
}

# Copy src/ directory
Write-Host "  ⏳ Copying src/ (Python scripts)..."
Copy-Item "$REPO_ROOT\src\*" "$HDD_DataPath\src\" -Recurse -Force
Write-Host "  ✓ src/ copied"

# Copy docs
Write-Host "  ⏳ Copying docs/..."
if (Test-Path "$REPO_ROOT\docs") {
    Copy-Item "$REPO_ROOT\docs\*" "$HDD_DataPath\docs\" -Recurse -Force
    Write-Host "  ✓ docs/ copied"
}

# Copy notebooks (optional)
if (Test-Path "$REPO_ROOT\notebooks") {
    Write-Host "  ⏳ Copying notebooks/ (optional)..."
    Copy-Item "$REPO_ROOT\notebooks\*" "$HDD_DataPath\notebooks\" -Recurse -Force
    Write-Host "  ✓ notebooks/ copied"
}

# ============================================================================
# STEP 4: Copy processed data (train.csv, test.csv, etc.)
# ============================================================================

Write-Host ""
Write-Host "Step 4/5: Copying processed training data..."

if (Test-Path "$REPO_ROOT\data\processed") {
    Write-Host "  ⏳ Copying data/processed/ (train.csv, test.csv, metadata)..."
    Copy-Item "$REPO_ROOT\data\processed\*" "$HDD_DataPath\data\processed\" -Recurse -Force

    $processed_size = (Get-ChildItem "$HDD_DataPath\data\processed" -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "  ✓ data/processed/ copied ($([Math]::Round($processed_size, 2)) GB)"
} else {
    Write-Host "  ⚠ data/processed/ not found (will generate on new PC)"
}

# ============================================================================
# STEP 5: Copy trained models (optional but useful)
# ============================================================================

Write-Host ""
Write-Host "Step 5/5: Copying pre-trained models..."

if (Test-Path "$REPO_ROOT\models") {
    $model_files = Get-ChildItem "$REPO_ROOT\models" -File -ErrorAction SilentlyContinue
    if ($model_files) {
        Write-Host "  ⏳ Copying models/ (pre-trained weights)..."
        Copy-Item "$REPO_ROOT\models\*" "$HDD_DataPath\models\" -Force
        Write-Host "  ✓ Models copied:"
        foreach ($file in $model_files) {
            $size = $file.Length / 1MB
            Write-Host "    - $($file.Name) ($([Math]::Round($size, 1)) MB)"
        }
    } else {
        Write-Host "  ℹ No pre-trained models found (you'll train new ones)"
    }
}

# ============================================================================
# FINAL VERIFICATION & SUMMARY
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║  ✅ MIGRATION COMPLETE!                                       ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

$total_size = (Get-ChildItem "$HDD_DataPath" -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "📊 Summary:"
Write-Host "  Total data on HDD: $([Math]::Round($total_size, 2)) GB"
Write-Host "  Location: $HDD_DataPath"
Write-Host ""

Write-Host "📝 Files copied:"
Write-Host "  ✓ Source code (src/)"
Write-Host "  ✓ Configuration (requirements.txt, .gitignore, etc.)"
Write-Host "  ✓ Training data (data/processed/)"
Write-Host "  ✓ Pre-trained models (if available)"
Write-Host "  ✓ Documentation (docs/)"
Write-Host ""

Write-Host "🎯 Next steps on the NEW PC:"
Write-Host "  1. Connect this HDD to new PC via USB"
Write-Host "  2. Run: python -m pip install -r $HDD_DataPath\requirements.txt"
Write-Host "  3. Run: python $HDD_DataPath\src\train_dl_model.py --model transformer"
Write-Host ""

Write-Host "⚠️  IMPORTANT: The dataset (arg_cleaned_dataset) should already be"
Write-Host "   on the HDD or will be transferred separately (220 GB)."
Write-Host ""

Write-Host "✅ You're ready to move to the new PC!"
