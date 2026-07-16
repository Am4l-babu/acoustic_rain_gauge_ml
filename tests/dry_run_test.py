"""
DRY RUN: Test on January_2024 folder only (3,325 files, 3s clips).
Processes only the first 50 audio files to validate:
  - Timestamp extraction from filenames
  - Mechanical data loading & column detection
  - Audio alignment logic
  - Feature extraction
"""
import os
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from datetime import datetime, timedelta
import json
import re

SOURCE_DRIVE = Path(r"F:\arg_dataset_unzip")
DEST = Path(r"D:\arg_cleaned_dataset\_dry_run_january_2024")
DEST.mkdir(parents=True, exist_ok=True)

AUDIO_FOLDER = SOURCE_DRIVE / "January_2024_rainfall_audios_with_mech_data"
MECH_FOLDER  = SOURCE_DRIVE / "January_2024_rainfall_audios_with_mech_data"
MAX_FILES = 50  # Only process first 50 files for the test


def extract_timestamp_from_filename(filename):
    name = Path(filename).stem
    patterns = [
        r'(\d{4})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})',
        r'(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            g = match.groups()
            try:
                return datetime(int(g[0]), int(g[1]), int(g[2]),
                                int(g[3]), int(g[4]), int(g[5]))
            except Exception:
                continue
    return None


def find_mechanical_data(folder_path):
    for csv_file in folder_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip().str.lower()
            time_col = rain_col = None
            for col in df.columns:
                if ('time' in col or 'timestamp' in col) and time_col is None:
                    time_col = col
                if col in ('rainfall (mm)', 'rainfall') and rain_col is None:
                    rain_col = col
            if rain_col is None:
                for col in df.columns:
                    if 'payload' in col or 'measurement' in col:
                        rain_col = col; break
            if time_col and rain_col:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
                return {'dataframe': df, 'time_col': time_col, 'rain_col': rain_col, 'file': csv_file}
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")
    return None


print("=" * 60)
print("DRY RUN - January 2024 (first 50 files)")
print("=" * 60)

# 1. Load mechanical data
print("\n[1] Loading mechanical data...")
mech = find_mechanical_data(MECH_FOLDER)
if mech is None:
    print("  FAILED: no mechanical data found")
    exit(1)
print(f"  File   : {mech['file'].name}")
print(f"  Rows   : {len(mech['dataframe'])}")
print(f"  Cols   : time={mech['time_col']}, rain={mech['rain_col']}")
print(f"  Range  : {mech['dataframe'][mech['time_col']].min()} -> {mech['dataframe'][mech['time_col']].max()}")
print(f"\n  First 5 rows:")
print(mech['dataframe'][[mech['time_col'], mech['rain_col']]].head().to_string(index=False))

# 2. Collect audio files
print("\n[2] Collecting audio files...")
audio_files = sorted(AUDIO_FOLDER.rglob("*.wav"))
print(f"  Total WAV files: {len(audio_files)}")
print(f"  Sample filenames:")
for f in audio_files[:5]:
    print(f"    {f.name}")

# 3. Test timestamp extraction on first 10 filenames
print("\n[3] Testing timestamp extraction...")
ts_ok = ts_fail = 0
for af in audio_files[:20]:
    ts = extract_timestamp_from_filename(af.name)
    status = "OK" if ts else "FAIL"
    if ts: ts_ok += 1
    else:  ts_fail += 1
    print(f"  [{status}] {af.name} -> {ts}")
print(f"\n  Parsed: {ts_ok}/20, Failed: {ts_fail}/20")

# 4. Test alignment on first MAX_FILES
print(f"\n[4] Testing alignment + feature extraction (first {MAX_FILES} files)...")
mech_df  = mech['dataframe']
time_col = mech['time_col']
rain_col = mech['rain_col']

records = []
for audio_file in audio_files[:MAX_FILES]:
    audio_time = extract_timestamp_from_filename(audio_file.name)
    if audio_time is None:
        print(f"  SKIP (no ts): {audio_file.name}")
        continue

    time_diffs  = (mech_df[time_col] - audio_time).abs()
    closest_idx = time_diffs.idxmin()
    min_diff    = time_diffs.min()

    if min_diff < timedelta(minutes=2):
        raw = str(mech_df.loc[closest_idx, rain_col]).replace('mm', '').strip()
        try:
            rainfall_mm = float(raw)
        except ValueError:
            rainfall_mm = 0.0
        aligned = True
    else:
        rainfall_mm = 0.0
        aligned = False

    # Feature extraction
    try:
        y, sr = librosa.load(str(audio_file), sr=16000, duration=None)
        rms  = float(np.sqrt(np.mean(y**2)))
        peak = float(np.max(np.abs(y)))
        records.append({
            'filename': audio_file.name,
            'timestamp': audio_time,
            'min_diff_sec': min_diff.total_seconds(),
            'aligned': aligned,
            'rainfall_mm': rainfall_mm,
            'duration_sec': round(len(y)/sr, 2),
            'rms': round(rms, 6),
            'peak': round(peak, 6),
        })
    except Exception as e:
        print(f"  AUDIO ERROR {audio_file.name}: {e}")

df = pd.DataFrame(records)
print(f"\n  Processed: {len(df)} files")
print(f"  Aligned  : {df['aligned'].sum()}/{len(df)}")
print(f"  Rainy    : {(df['rainfall_mm'] > 0).sum()}")
print(f"  Dry      : {(df['rainfall_mm'] == 0).sum()}")
print(f"  Max gap  : {df['min_diff_sec'].max():.1f}s")
print(f"  Durations: {df['duration_sec'].unique()}")
print(f"\n  Sample records:")
print(df[['filename','timestamp','aligned','rainfall_mm','duration_sec']].head(10).to_string(index=False))

# 5. Save dry-run output
df.to_csv(DEST / "dry_run_output.csv", index=False)
print(f"\n[5] Dry-run CSV saved to: {DEST / 'dry_run_output.csv'}")
print("\n" + "=" * 60)
print("DRY RUN COMPLETE")
print("=" * 60)
