# Troubleshooting: Common Issues & Solutions

## Setup Issues

### ❌ "HDD not found at E:\acoustic_rain_gauge_ml"

**Cause:** HDD connected but at wrong drive letter

**Solution:**
1. Open File Explorer
2. Look for the new HDD drive (might be F:, G:, etc., not E:)
3. Note the correct drive letter
4. Re-run setup with correct letter:
   ```powershell
   .\NEW_PC_SETUP.ps1 -HDD_Drive "F:" -LocalPath "D:\acoustic_rain_gauge_ml"
   ```

---

### ❌ "Python not found"

**Cause:** Python not installed or not in PATH

**Solution:**
1. Install Python 3.10+ from https://www.python.org/downloads/
2. **CRITICAL:** During installation, check the box "Add Python to PATH"
3. After installation, restart PowerShell
4. Test: `python --version`
5. Re-run setup script

---

### ❌ "venv not activated" (no `(.venv)` in prompt)

**Cause:** Virtual environment not activated in current PowerShell session

**Solution:**
```powershell
cd D:\acoustic_rain_gauge_ml
.\.venv\Scripts\Activate.ps1
```

If you see an error like "PowerShell does not allow script execution", run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry the activate command.

---

### ❌ "Module not found: torch" / "No module named pandas"

**Cause:** venv not activated, OR dependencies not installed

**Solution:**
```powershell
# Make sure venv is activated (should show (.venv) in prompt)
cd D:\acoustic_rain_gauge_ml
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Training Issues

### ❌ "CUDA out of memory" during training

**Cause:** Batch size too large for GPU memory

**Solution:**
```powershell
# Reduce batch size
python src/train_dl_model.py --model transformer --batch_size 8

# Or even smaller if needed
python src/train_dl_model.py --model transformer --batch_size 4
```

If using CPU (no GPU), it may still be slow with large batch sizes. Start with batch_size=4.

---

### ❌ "CUDA not available" / GPU not detected

**Cause:** NVIDIA GPU not detected, or PyTorch installed without GPU support

**Solution 1: Check what's installed**
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Solution 2: If False, reinstall PyTorch with GPU support**
```powershell
# Reinstall CUDA version
pip uninstall torch torchaudio -y
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**Note:** Training still works on CPU, just slower (10-30 min instead of 1-2 hours)

---

### ❌ "ModuleNotFoundError: No module named 'librosa'" during feature extraction

**Cause:** Dependencies not installed

**Solution:**
```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Reinstall all dependencies
pip install -r requirements.txt

# Then retry
python src/advanced_feature_extraction.py
```

---

### ❌ Training seems stuck / very slow

**Cause:** Could be multiple things (CPU bottleneck, disk I/O, wrong dataset path)

**Solution:**
1. Check CPU/GPU usage:
   ```powershell
   Get-Process python | Select-Object CPU, Memory, ProcessName
   nvidia-smi -l 1  # If GPU available
   ```

2. If CPU is low (<5%) or GPU is 0%, something's wrong:
   - Check dataset path: `ls E:\arg_cleaned_dataset\`
   - Check data files: `ls D:\acoustic_rain_gauge_ml\data\processed\`
   - Check disk space: `Get-Volume`

3. If disk is full, clear space or move data temporarily

---

## Feature Extraction Issues

### ❌ "advanced_features_dataset.csv not found"

**Cause:** Feature extraction still running OR crashed

**Solution 1: Check if still running**
```powershell
Get-Process python  # See if python process is active
ls -lh D:\advanced_features\  # Check output file size
```

**Solution 2: If crashed, restart it**
```powershell
cd D:\acoustic_rain_gauge_ml
.\.venv\Scripts\Activate.ps1
python src/advanced_feature_extraction.py
# It will skip already-processed files and continue
```

---

### ❌ Feature extraction very slow (< 100 files/min)

**Cause:** Possibly dataset on slow HDD, or I/O bottleneck

**Solution:**
1. Check CPU usage: should be 80-100%
   ```powershell
   Get-Process python | Select-Object CPU, Memory
   ```

2. If CPU is low, bottleneck is disk:
   - Try moving dataset to faster drive temporarily
   - Or accept slower speed (it will still finish, just overnight)

3. If CPU is high but still slow, it's normal (audio processing is just slow)

---

### ❌ "Dataset folder not found" during feature extraction

**Cause:** Dataset path wrong or dataset not on HDD

**Solution:**
1. Verify dataset exists:
   ```powershell
   ls E:\arg_cleaned_dataset\  # Check if this works
   # If not, find correct path and note drive letter
   ```

2. Edit `src/advanced_feature_extraction.py` line 14:
   ```python
   DATA_DIR = Path(r'E:\arg_cleaned_dataset')  # Adjust path if needed
   ```

3. Retry feature extraction

---

## Disk Space Issues

### ❌ "No space left on device" during training/merging

**Cause:** Disk full

**Solution 1: Check free space**
```powershell
Get-Volume | Select-Object DriveLetter, SizeGB, SizeRemainingGB
```

**Solution 2: Free up space**
- Delete old datasets or temporary files
- Move .venv to another drive temporarily
- Move advanced_features CSV to HDD after merging

**Solution 3: Use HDD for temporary files**
```python
# In merge script, change:
# advanced_features_dataset.csv location from D:\ to E:\
```

---

## Data Path Issues

### ❌ "FileNotFoundError: train.csv" or "test.csv"

**Cause:** Data files not where expected

**Solution:**
1. Check if files exist:
   ```powershell
   ls D:\acoustic_rain_gauge_ml\data\processed\
   ```

2. If files missing, regenerate them:
   ```powershell
   cd D:\acoustic_rain_gauge_ml
   .\.venv\Scripts\Activate.ps1
   python src/feature_extraction.py  # ~15-20 min
   ```

3. Verify output:
   ```powershell
   ls D:\acoustic_rain_gauge_ml\data\processed\
   # Should show: train.csv, test.csv
   ```

---

## GPU-Specific Issues

### ❌ "NVIDIA driver issue" or "CUDA mismatch"

**Cause:** Old NVIDIA drivers or CUDA version mismatch

**Solution:**
1. Update NVIDIA drivers from nvidia.com
2. Check driver version:
   ```powershell
   nvidia-smi
   ```
   Should show driver 550+ for CUDA 12.4

3. Reinstall PyTorch:
   ```powershell
   pip uninstall torch torchaudio -y
   pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
   ```

---

### ❌ CUDA works but training is still slow

**Cause:** GPU not actually being used (might be CPU-bound data loading)

**Solution 1: Check GPU usage during training**
```powershell
nvidia-smi -l 1  # Update every 1 second
# GPU-Util should be 90%+, Memory should be high
```

**Solution 2: If GPU is idle, issue is data loading**
- Batch size might be too small (less GPU work)
- Try increasing batch size:
  ```powershell
  python src/train_dl_model.py --batch_size 64
  ```

---

## Performance Issues

### ❌ "Training took 8 hours, expected 2 hours"

**Cause:** Running on CPU instead of GPU

**Solution:**
1. Verify GPU is being used:
   ```powershell
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

2. If False, reinstall PyTorch with CUDA support (see GPU issues section above)

3. During next training, monitor:
   ```powershell
   nvidia-smi -l 1
   ```
   GPU memory should be used, GPU-Util should be 90%+

---

### ❌ Training loss not decreasing / model not learning

**Cause:** Could be data issue, learning rate, or initialization

**Solution:**
1. First verify baseline works:
   ```powershell
   python src/train_model.py  # XGBoost should give R²≈0.155
   ```

2. If baseline fails, data issue:
   - Verify files: `ls D:\acoustic_rain_gauge_ml\data\processed\`
   - Verify format: `python -c "import pandas as pd; df = pd.read_csv('data/processed/train.csv'); print(df.head())"`

3. If baseline works but Transformer doesn't:
   - Try different learning rate
   - Check if features are normalized properly
   - Try running with CPU first (slower but easier to debug)

---

## File Permission Issues

### ❌ "Access denied" when creating files

**Cause:** Permission issue on SSD or venv not properly created

**Solution:**
1. Run PowerShell as Administrator
2. Recreate virtual environment:
   ```powershell
   rm -r D:\acoustic_rain_gauge_ml\.venv
   python -m venv D:\acoustic_rain_gauge_ml\.venv
   D:\acoustic_rain_gauge_ml\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

---

## Help! Nothing Above Helped

### Get Debugging Information

Before asking for help, collect this info:

```powershell
# System info
python --version
pip --version
nvidia-smi  # (if GPU)

# Check environment
cd D:\acoustic_rain_gauge_ml
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check data
ls D:\acoustic_rain_gauge_ml\data\processed\
ls E:\arg_cleaned_dataset\

# Copy error message EXACTLY (last 50 lines of error)
```

Share this info plus the exact error message and it will be easier to diagnose!

---

## Still Stuck?

1. **Read the error message carefully** — most are self-explanatory
2. **Check STEP_BY_STEP_GUIDE.txt** for exact commands
3. **Verify each prerequisite** before moving to next step
4. **Try simpler option first** — Option A (XGBoost) is easiest to debug
5. **Check disk space** — many mysterious errors are actually disk full
6. **Try on CPU** — if GPU issues, fall back to CPU-only training for debugging

Good luck! 🚀
