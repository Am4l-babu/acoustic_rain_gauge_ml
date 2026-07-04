# 📚 Index: New PC Setup Folder

Complete collection of guides, scripts, and documentation for migrating the acoustic rain gauge ML project to a new PC.

---

## 🚀 Quick Links

| Need | File | Time |
|------|------|------|
| **Overview** | [README.md](README.md) | 5 min |
| **Commands** | [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt) | 5 min |
| **Explanation** | [SETUP_OVERVIEW.md](SETUP_OVERVIEW.md) | 10 min |
| **Choose Option** | [ALL_OPTIONS_COMPARISON.md](ALL_OPTIONS_COMPARISON.md) | 10 min |
| **Option C Only** | [OPTION_C_COMMANDS.txt](OPTION_C_COMMANDS.txt) | 5 min |
| **Help** | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Reference |
| **This Index** | [FILE_MANIFEST.txt](FILE_MANIFEST.txt) | 5 min |

---

## 📂 What's in This Folder

```
docs/NEW_PC_SETUP/
├── README.md                      ← Start here (overview)
├── INDEX.md                       ← This file
├── SETUP_OVERVIEW.md              ← Understand the process
├── STEP_BY_STEP_GUIDE.txt         ← Copy-paste commands
├── ALL_OPTIONS_COMPARISON.md      ← A vs B vs C
├── OPTION_C_COMMANDS.txt          ← Advanced features guide
├── TROUBLESHOOTING.md             ← Fix common issues
├── FILE_MANIFEST.txt              ← File descriptions
└── NEW_PC_SETUP.ps1               ← Run on new PC ⭐
```

---

## 🎯 Quick Start (3 steps)

### Step 1: Prepare Current PC (15 min)
**Location:** D:\acoustic_rain_gauge_ml\
```powershell
.\migrate_to_hdd.ps1 -TargetDrive "E:" -HDD_DataPath "E:\acoustic_rain_gauge_ml"
```

### Step 2: Transfer Hardware
Disconnect HDD, connect to new PC

### Step 3: Setup New PC (20 min)
**Location:** Wherever you saved NEW_PC_SETUP.ps1
```powershell
.\NEW_PC_SETUP.ps1 -HDD_Drive "E:" -LocalPath "D:\acoustic_rain_gauge_ml"
```

**Done!** Now train using commands from [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)

---

## 📊 Three Training Paths

| Option | Time | R² | Best For |
|--------|------|-----|----------|
| **A** | 5 min | 0.155 | Quick validation |
| **B** | 1-2 hours | 0.30 | Good results |
| **C** ⭐ | 16-23 hours | 0.40-0.45 | Best results |

See [ALL_OPTIONS_COMPARISON.md](ALL_OPTIONS_COMPARISON.md) for details.

---

## 🔧 How to Use These Files

### For a Quick Start:
1. Copy [NEW_PC_SETUP.ps1](NEW_PC_SETUP.ps1) to new PC
2. Copy [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt) for reference
3. Follow the commands

### For Understanding:
1. Read [README.md](README.md) (5 min)
2. Read [SETUP_OVERVIEW.md](SETUP_OVERVIEW.md) (10 min)
3. Read [ALL_OPTIONS_COMPARISON.md](ALL_OPTIONS_COMPARISON.md) (10 min)
4. Then run commands from [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)

### For Option C (Advanced Features):
1. Follow [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt) STEP 4C
2. Keep [OPTION_C_COMMANDS.txt](OPTION_C_COMMANDS.txt) open for reference
3. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if issues

### If Something Breaks:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for your error
2. Or re-read the relevant section in [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)

---

## 📋 File Descriptions

### Setup & Overview
- **[README.md](README.md)** — Overview & navigation hub
- **[INDEX.md](INDEX.md)** — This file (file index)
- **[FILE_MANIFEST.txt](FILE_MANIFEST.txt)** — Detailed file inventory

### Process Documentation
- **[SETUP_OVERVIEW.md](SETUP_OVERVIEW.md)** — Deep dive: why, how, what
- **[STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)** — All commands organized
- **[ALL_OPTIONS_COMPARISON.md](ALL_OPTIONS_COMPARISON.md)** — A vs B vs C comparison

### Training Guides
- **[OPTION_C_COMMANDS.txt](OPTION_C_COMMANDS.txt)** — Commands for advanced features path

### Support
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — 15+ common issues & fixes

### Scripts
- **[NEW_PC_SETUP.ps1](NEW_PC_SETUP.ps1)** — Automated setup (run on new PC)

---

## ⏱️ Timeline & Effort

| Phase | Time | Effort | What |
|-------|------|--------|------|
| **Prepare** | 15 min | Low | Run migrate_to_hdd.ps1 |
| **Setup** | 20 min | Low | Run NEW_PC_SETUP.ps1 |
| **Train A** | 5 min | Minimal | Quick test |
| **Train B** | 1-2 hr | Minimal | Good results |
| **Train C** | 14-20 hr extraction + 1-2 hr training | Hands-off | Best results |

---

## ✅ Verification Checklist

After each step, verify:

- [ ] **After migrate_to_hdd.ps1:**
  - `ls E:\acoustic_rain_gauge_ml\src\` shows files
  - `ls E:\arg_cleaned_dataset\` shows 19 folders

- [ ] **After NEW_PC_SETUP.ps1:**
  - `ls D:\acoustic_rain_gauge_ml\` shows project
  - `python -c "import torch; print(torch.cuda.is_available())"` works

- [ ] **After training:**
  - `cat docs\model_evaluation_report.json` shows results
  - R² score makes sense for your chosen option

---

## 🎯 Key Takeaways

✅ **Fully automated** — Scripts handle setup  
✅ **Three options** — Pick speed vs accuracy  
✅ **GPU optional** — Works on CPU too  
✅ **Dataset stays on HDD** — Only copies essential files  
✅ **Documentation included** — No guessing required  

---

## 📞 Need Help?

1. **"I don't understand the process"**
   → Read [SETUP_OVERVIEW.md](SETUP_OVERVIEW.md)

2. **"I just need commands"**
   → Use [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)

3. **"I need to choose between options"**
   → Read [ALL_OPTIONS_COMPARISON.md](ALL_OPTIONS_COMPARISON.md)

4. **"Something is broken"**
   → Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

5. **"Where's the migrate_to_hdd.ps1?"**
   → In project root: `D:\acoustic_rain_gauge_ml\migrate_to_hdd.ps1`

---

## 🗂️ Where Everything Goes

### On Current PC:
- ✅ Already: `D:\acoustic_rain_gauge_ml\migrate_to_hdd.ps1`

### On HDD (after migrate):
- ✅ `E:\acoustic_rain_gauge_ml\` — Project code
- ✅ `E:\arg_cleaned_dataset\` — Training data (220 GB)

### On New PC (after setup):
- ✅ `D:\acoustic_rain_gauge_ml\` — Project on SSD (fast)
- ✅ `E:\acoustic_rain_gauge_ml\` — Backup on HDD
- ✅ `E:\arg_cleaned_dataset\` — Dataset stays on HDD

---

## 🚀 Ready?

1. Start with [README.md](README.md) or [STEP_BY_STEP_GUIDE.txt](STEP_BY_STEP_GUIDE.txt)
2. Keep [FILE_MANIFEST.txt](FILE_MANIFEST.txt) nearby for reference
3. Use [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if needed
4. You've got this! 💪

---

**Created:** 2026-07-04  
**Status:** Complete & tested  
**Version:** 1.0

For more info on specific topics, see individual file descriptions above.
