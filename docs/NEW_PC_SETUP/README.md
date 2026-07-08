# 🖥️ New PC Setup & HDD Migration Guide

Complete step-by-step guide for migrating this acoustic rain gauge ML project to a new PC with better specs.

---

## 📋 Quick Navigation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[SETUP_OVERVIEW.md](SETUP_OVERVIEW.md)** | 📌 Start here — understand the full process | 5 min |
| **[STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)** | Copy-paste commands for entire workflow | 10 min |
| **[NEW_PC_SETUP.ps1](Windows/NEW_PC_SETUP.ps1)** | Automated setup script (run on new PC) | N/A |
| **[OPTION_C_ADVANCED_DL.md](OPTION_C_ADVANCED_DL.md)** | Deep learning + 63 advanced features guide | 20 min |
| **[OPTION_C_COMMANDS.txt](OPTION_C_COMMANDS.txt)** | Copy-paste commands for Option C only | 5 min |
| **[ALL_OPTIONS_COMPARISON.md](ALL_OPTIONS_COMPARISON.md)** | Compare all training paths (XGBoost/DL) | 10 min |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues & solutions | Reference |

---

## 🐧 New PC is Linux/Ubuntu instead of Windows?

Everything above (`Windows/NEW_PC_SETUP.ps1` etc.) is PowerShell-only. For an Ubuntu new PC, use the `Ubuntu/` folder instead — it covers the Stage 8 master feature extraction workflow specifically (not the full training pipeline, which is still Windows-only in this doc set):

```bash
cd /media/$USER/E-HDD/acoustic_rain_gauge_ml   # or wherever the HDD mounted
bash docs/NEW_PC_SETUP/Ubuntu/CHECK_SPECS.sh       # RAM/CPU/GPU/disk check + safe --workers suggestion
bash docs/NEW_PC_SETUP/Ubuntu/RUN_EXTRACTION.sh    # pip install -> smoke test -> validate -> full run
```

---

## 🚀 Quick Start (5 Minutes)

### On Current PC:
```powershell
cd D:\acoustic_rain_gauge_ml
.\migrate_to_hdd.ps1 -TargetDrive "E:" -HDD_DataPath "E:\acoustic_rain_gauge_ml"
# Wait for: "✅ MIGRATION COMPLETE!"
```

### On New PC (after connecting HDD):
```powershell
.\Windows\NEW_PC_SETUP.ps1 -HDD_Drive "E:" -LocalPath "D:\acoustic_rain_gauge_ml"
# Wait for: "✅ SETUP COMPLETE! READY TO TRAIN"
```

### Then Train:
**Option A (Fast):** XGBoost in 5 minutes
```powershell
python src/train_model.py && python src/evaluate_model.py
```

**Option B (Good):** Transformer in 1-2 hours
```powershell
python src/train_dl_model.py --model transformer --epochs 50
```

**Option C (Best):** Advanced features + Transformer in 16-23 hours
```powershell
python src/advanced_feature_extraction.py  # 14-20 hours (leave running)
# Then merge and train...
python src/train_dl_model.py --model transformer --epochs 50
```

---

## 📊 Three Training Options

| Option | Features | Model | Time | Expected R² |
|--------|----------|-------|------|-------------|
| **A** | MFCC only | XGBoost | 5 min | 0.155 |
| **B** | MFCC only | Transformer | 1-2 hours | 0.30 |
| **C** ⭐ | MFCC + 63 Advanced | Transformer | 16-23 hours | 0.40-0.45 |

---

## 📁 What's in This Folder

```
docs/NEW_PC_SETUP/
├── README.md                        ← You are here
├── SETUP_OVERVIEW.md                ← Start here (5 min read)
├── STEP_BY_STEP_GUIDE.txt           ← All commands
├── Windows/NEW_PC_SETUP.ps1                 ← Run this on new PC
├── OPTION_C_ADVANCED_DL.md          ← For advanced features path
├── OPTION_C_COMMANDS.txt            ← Commands for Option C
├── ALL_OPTIONS_COMPARISON.md        ← Compare all options
├── TROUBLESHOOTING.md               ← Fix common issues
└── FILE_MANIFEST.txt                ← This folder's contents
```

---

## ⚡ System Requirements

### New PC Needs:
- **OS:** Windows 10/11
- **Python:** 3.10 or higher
- **Disk space:** 50+ GB free on SSD
- **GPU:** Optional but recommended (NVIDIA RTX 3060+ for fast training)

### Data Location:
- **220 GB dataset:** Stays on external HDD
- **Project files:** Copy to new PC's SSD (for speed)
- **Models:** Trained on local SSD, backed up on HDD

---

## 🎯 Recommended Workflow

### Friday Evening (30 minutes):
1. Connect HDD to new PC
2. Run `NEW_PC_SETUP.ps1` (20 min)
3. Start feature extraction if doing Option C (leave running overnight)

### Saturday Morning:
1. Check feature extraction status
2. If Option C: run training (1-2 hours)
3. Review results

---

## ✨ Key Features of This Setup

✅ **Fully Automated** — Setup script handles all dependencies  
✅ **GPU-Optimized** — Detects NVIDIA GPU and installs CUDA PyTorch  
✅ **Fast Training** — GPU acceleration for deep learning models  
✅ **Large Dataset** — Works with 220GB dataset on HDD  
✅ **Validated** — All paths and dependencies verified  
✅ **Documented** — Multiple guides for different experience levels  

---

## 🆘 Need Help?

1. **Setup issues?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Want advanced features?** → [OPTION_C_ADVANCED_DL.md](OPTION_C_ADVANCED_DL.md)
3. **Just want commands?** → [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)
4. **Choosing options?** → [ALL_OPTIONS_COMPARISON.md](ALL_OPTIONS_COMPARISON.md)

---

## 📊 Expected Results After Full Training (Option C)

```
Baseline (XGBoost):     R² = 0.155
With Option C setup:    R² = 0.40-0.45 ✨
Improvement:            2.6x better! 🎯
```

---

**Created:** 2026-07-04  
**Status:** Ready for new PC migration  
**Last Updated:** Latest guides included
