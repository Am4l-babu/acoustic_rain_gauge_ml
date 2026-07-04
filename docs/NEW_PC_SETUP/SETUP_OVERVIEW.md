# Setup Overview: Current PC → New PC

This guide explains the **why** and **how** of migrating this acoustic rain gauge ML project to a new PC with better specs.

---

## 🎯 What You're Doing

You have:
- **Current PC:** Original setup with project code
- **New PC:** Better specs (more RAM, faster GPU, better CPU)
- **External HDD:** 220GB dataset + backup project files
- **Goal:** Run advanced ML training on the new PC

**Solution:** Two-script migration that takes ~35 minutes hands-on time.

---

## 📍 The Migration Strategy

### Phase 1: Current PC (15 minutes)
```
migrate_to_hdd.ps1 runs on CURRENT PC
  ↓
Copies essential files to HDD
  ├─ src/ (training scripts)
  ├─ data/processed/ (train.csv, test.csv)
  ├─ models/ (pre-trained models)
  ├─ docs/ (documentation)
  ├─ requirements.txt
  └─ config files
  ↓
HDD now has: Everything except .venv and .git
```

### Phase 2: Transfer (Physical)
```
Disconnect HDD from current PC
  ↓
Connect HDD to new PC
```

### Phase 3: New PC (20 minutes)
```
NEW_PC_SETUP.ps1 runs on NEW PC
  ↓
Step 1: Verifies HDD connection
Step 2: Copies project to local SSD (for speed)
Step 3: Checks Python installation
Step 4: Creates fresh virtual environment
Step 5: Installs all dependencies (including GPU support)
  ↓
New PC ready to train immediately!
```

---

## ⏱️ Why This Approach?

### ✅ Advantages
- **Fast setup:** Only 20 minutes on new PC
- **GPU-optimized:** Automatically detects NVIDIA GPU and installs CUDA PyTorch
- **No manual configuration:** Scripts handle all paths
- **Dataset flexibility:** 220GB dataset stays on HDD, only needed files on SSD
- **Automated:** No manual file copying or pip commands
- **Reproducible:** Same environment as old PC

### ❌ What We're NOT Doing
- ❌ Not copying entire .git folder (unnecessary, project was cloned)
- ❌ Not copying .venv folder (too large, recreated from requirements.txt)
- ❌ Not manual pip install (all automated by setup script)
- ❌ Not pre-copying dataset (already on HDD)

---

## 📊 File Inventory

### Files That Get Copied (Essential)
✅ `src/` — Python training scripts
✅ `data/processed/` — Training/test data (CSVs)
✅ `models/` — Pre-trained models
✅ `docs/` — Documentation
✅ `requirements.txt` — Python dependencies
✅ `.gitignore`, `README.md` — Configuration
✅ `notebooks/` — Analysis notebooks (optional)

**Size:** ~0.7 GB (fits easily on HDD with dataset)

### Files NOT Copied (Recreated)
❌ `.venv/` — Virtual environment (too large, recreated on new PC)
❌ `.git/` — Git history (not needed for training)
❌ `__pycache__/` — Python cache (regenerated)

---

## 🔧 What The Scripts Do

### `migrate_to_hdd.ps1` (Current PC)
```powershell
1. Validates HDD path exists
2. Creates E:\acoustic_rain_gauge_ml\ folder
3. Copies src/ → HDD
4. Copies data/processed/ → HDD
5. Copies models/ → HDD
6. Copies requirements.txt → HDD
7. Verifies 220GB dataset already on HDD
8. Done! (~15 min)
```

### `NEW_PC_SETUP.ps1` (New PC)
```powershell
1. Verifies HDD connected with files
2. Copies project to D:\acoustic_rain_gauge_ml (local SSD)
3. Verifies Python 3.10+ installed
4. Creates .venv in new location
5. Installs pip packages from requirements.txt
6. Detects GPU (nvidia-smi)
7. If GPU found: installs PyTorch with CUDA 12.4
8. If no GPU: installs CPU version
9. Verifies CUDA works: torch.cuda.is_available()
10. Done! (~20 min, most is dependency install)
```

---

## 💾 Disk Space Breakdown

### Current PC
```
Source files to migrate: 0.7 GB
↓
Copied to HDD: 0.7 GB (in E:\acoustic_rain_gauge_ml\)
Dataset stays: 220 GB (already on HDD)
```

### New PC After Setup
```
Local SSD (D:\):
  ├─ acoustic_rain_gauge_ml/: 0.7 GB (project)
  ├─ .venv/: 1-2 GB (Python packages)
  └─ Working space: 10-20 GB (during training)
  ────────────────
  Total: ~30-50 GB (you need 50+ GB free)

HDD (E:\):
  ├─ acoustic_rain_gauge_ml/: 0.7 GB (backup copy)
  ├─ arg_cleaned_dataset/: 220 GB (training data)
  └─ advanced_features/: 2-3 GB (if doing Option C)
  ────────────────
  Total: 222+ GB
```

---

## 🚀 Training Paths After Setup

Once new PC is ready, you choose:

### **Path A: Quick Validation (5 min)**
```
Train XGBoost on existing MFCC features
Expected R²: 0.155 (baseline)
Use case: Verify setup works
```

### **Path B: Good Results (1-2 hours)**
```
Train Transformer on existing MFCC features
Expected R²: 0.30-0.31
Use case: Good accuracy, reasonable time
```

### **Path C: Best Results (16-23 hours)** ⭐
```
Extract 63 advanced audio features (14-20 hours)
  ├─ Teager Energy (raindrop detection)
  ├─ Wavelet decomposition (multi-scale patterns)
  ├─ Spectral flux (frequency changes)
  ├─ Histogram packets (temporal patterns)
  └─ Entropy (signal complexity)

Merge with existing features (5 min)

Train Transformer on all 103 features (1-2 hours)
Expected R²: 0.40-0.45 (2.6x better!)
Use case: Best possible accuracy, can run overnight
```

---

## 🎯 Why Advanced Features Take So Long?

For **760,000 audio clips**:

```
Per clip:
  • Load 10-second WAV file:        ~0.05 sec
  • Extract MFCC:                   ~0.03 sec
  • Extract Teager Energy:          ~0.02 sec
  • Extract Wavelet (6 levels):     ~0.03 sec
  • Extract Spectral Flux:          ~0.02 sec
  • Extract Entropy features:       ~0.02 sec
  ─────────────────────────────────────────
  Total per clip:                   ~0.17 sec

For all clips:
  760,000 × 0.17 sec = 129,200 sec = 36 hours theoretical
  (With optimizations: 14-20 hours in practice)
```

**Why so long?**
- ❌ **Not GPU-acceleratable:** Audio processing libraries (librosa, scipy) are CPU-only
- ❌ **Sequential algorithms:** Each clip must be processed one-by-one
- ❌ **Complex math:** Wavelet decomposition, entropy calculation per clip

**But training is fast!**
- ✅ **GPU-accelerated:** Once features are extracted, Transformer training uses GPU
- ✅ **Batched:** GPU processes 32 samples at once, not one-by-one
- ✅ **Just learning:** Features already computed, no more processing needed

---

## ✅ Verification Checklist

### Before Running Scripts
- [ ] Python 3.10+ installed (current PC already has it)
- [ ] HDD detected and accessible
- [ ] 50+ GB free space on new PC's SSD
- [ ] GPU drivers up-to-date (if GPU available)

### After `migrate_to_hdd.ps1`
- [ ] File: `E:\acoustic_rain_gauge_ml\requirements.txt` exists
- [ ] File: `E:\acoustic_rain_gauge_ml\src\train_model.py` exists
- [ ] Folder: `E:\acoustic_rain_gauge_ml\data\processed\` has CSVs
- [ ] Folder: `E:\arg_cleaned_dataset\` has 19 month folders

### After `NEW_PC_SETUP.ps1`
- [ ] Folder: `D:\acoustic_rain_gauge_ml\` exists
- [ ] File: `D:\acoustic_rain_gauge_ml\.venv\Scripts\python.exe` exists
- [ ] Command works: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] GPU available: Output shows `True` (or `False` for CPU)

---

## 🚨 If Something Goes Wrong

1. **"HDD not found"** → Check drive letter (might be F:, not E:)
2. **"Python not found"** → Install from python.org, check "Add to PATH"
3. **"venv already exists"** → Script handles this, no need to recreate
4. **"GPU not detected"** → Still works, but training slower (CPU mode)
5. **"CUDA out of memory"** → Reduce batch_size in training command

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

---

## 📞 Quick Command Reference

```powershell
# Current PC: Copy to HDD
cd D:\acoustic_rain_gauge_ml
.\migrate_to_hdd.ps1 -TargetDrive "E:" -HDD_DataPath "E:\acoustic_rain_gauge_ml"

# New PC: Setup everything
.\NEW_PC_SETUP.ps1 -HDD_Drive "E:" -LocalPath "D:\acoustic_rain_gauge_ml"

# New PC: Activate environment (always needed!)
cd D:\acoustic_rain_gauge_ml
.\.venv\Scripts\Activate.ps1

# Option A: Quick test (5 min)
python src/train_model.py
python src/evaluate_model.py

# Option B: Transformer (1-2 hours)
python src/train_dl_model.py --model transformer --epochs 50

# Option C: Advanced features (16-23 hours)
python src/advanced_feature_extraction.py  # 14-20 hours
# (after done) python ... merge features ...
python src/train_dl_model.py --model transformer --epochs 50
```

---

## 🎓 Understanding the Timeline

**Why 20+ hours if GPU is fast?**

The 20 hours breaks down as:
- ⏱️ **14-20 hours** = Feature extraction (CPU-bound, can't use GPU)
- ⏱️ **1-2 hours** = Model training (GPU-accelerated, very fast)
- ⏱️ **5 min** = Merging data
- ⏱️ **5 min** = Evaluation

Feature extraction is slow because it's sequential CPU work. Training is fast because it's parallel GPU work.

**Realistic Schedule:**
```
Friday 5:00 PM:  Run NEW_PC_SETUP.ps1 (20 min) ✓
Friday 5:30 PM:  Start feature extraction (leave running overnight)
Saturday 10:00 AM: Feature extraction done, run training (1-2 hours)
Saturday 12:00 PM: Results ready! 🎉
```

---

## 🎯 Next Steps

1. **Read:** [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt) — copy-paste commands
2. **On current PC:** Run `migrate_to_hdd.ps1`
3. **Connect HDD** to new PC
4. **On new PC:** Run `NEW_PC_SETUP.ps1`
5. **Choose your path:** A, B, or C training option
6. **Train:** Follow [OPTION_C_COMMANDS.txt](OPTION_C_COMMANDS.txt) if doing advanced features

---

**Ready?** Start with [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)! 🚀
