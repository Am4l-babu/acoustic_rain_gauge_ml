"""
Acoustic Rainfall Dataset Cleaning & Feature Extraction
========================================================
Processes paired (audio, mechanical) folders from the source drive,
extracts acoustic features in parallel (ProcessPoolExecutor), aligns
each recording to the nearest mechanical rainfall reading, and writes
per-folder CSVs + a global summary JSON.

Architecture
------------
  Main process  : timestamp extraction + mechanical alignment (fast, I/O bound)
  Worker pool   : librosa feature extraction per WAV file (slow, CPU bound)

Run:
    python src/data_cleaning.py
"""

import os
import shutil
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
SOURCE_DRIVE      = Path(r"F:\arg_dataset_unzip")
DESTINATION_DRIVE = Path(r"D:\arg_cleaned_dataset")
NUM_WORKERS       = 16        # workers for parallel audio processing
CHUNK_SIZE        = 32        # files per worker chunk
DELETE_PROCESSED_FOLDERS = False  # Set True AFTER verifying output

# ============================================================
# FOLDER PAIRS  (audio_folder_name, mechanical_folder_name)
# Pairs where both names are identical = combined old-style folder
# ============================================================
FOLDER_PAIRS = [
    # Old-style combined (audio + CSV in same tree)
    ("April_2024_rainfall_audios_with_mech_data",       "April_2024_rainfall_audios_with_mech_data"),
    ("December_2023_rainfall_audios_with_mech_data",    "December_2023_rainfall_audios_with_mech_data"),
    ("January_2024_rainfall_audios_with_mech_data",     "January_2024_rainfall_audios_with_mech_data"),
    ("July_2024_rainfall_audios_with_mech_data",        "July_2024_rainfall_audios_with_mech_data"),
    ("May_2024_rainfall_audios_with_mechanical_data",   "May_2024_rainfall_audios_with_mechanical_data"),
    ("November_2023_rainfall_audios_with_mech_data",    "November_2023_rainfall_audios_with_mech_data"),
    ("September_2024_rainfall_audios_with_mech_data",   "September_2024_rainfall_audios_with_mech_data"),
    # New-style split (separate audio / mechanical folders)
    ("August_2025_Rain_Data",        "August_2025_mechanical_rainfall_data"),
    ("December_2024_rain_data",      "December_2024_mechanical_rainfall_data"),
    ("Feb_to_Mar_2026_rainfall_data","Feb_to_March_2026_mechanical_rainfall_data"),
    ("January_2025_rainfall_data",   "January_2025_mechanical_rainfall_data"),
    ("June_2025_Rainfall_data",      "June_2025_mechanical_rainfall_data"),
    ("June_2026_rainfall_data",      "June_2026_mechanical_rainfall_data"),
    ("June_2026_rainfall_data_2",    "June_2026_mechanical_rainfall_data"),
    ("May_2025_Rainfall_Data",       "May_2025_mechanical_rainfall_data"),
    ("May_2026_rainfall_data",       "May_2026_mechanical_rainfall_data"),
    ("November_2024_Rainfall_Data",  "November_2024_mechanical_rainfall_data"),
    ("October_2025_rainfall_data",   "October_2025_mechanical_rainfall_data"),
    ("September_2025_rainfall_data", "September_2025_mechanical_rainfall_data"),
]

# ============================================================
# WORKER FUNCTION  (must be top-level for pickling on Windows)
# ============================================================
def _extract_features(audio_path_str: str) -> dict:
    """
    Called in a worker process.
    Loads a WAV file and returns acoustic features.
    Returns a dict with 'success': True/False.
    """
    try:
        y, sr = sf.read(audio_path_str)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        duration = len(y) / sr

        rms  = float(np.sqrt(np.mean(y ** 2)))
        peak = float(np.max(np.abs(y)))
        par  = peak / rms if rms > 0 else 0.0

        sc  = float(np.mean(librosa.feature.spectral_centroid(y=y,  sr=sr)))
        sbw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        sro = float(np.mean(librosa.feature.spectral_rolloff(y=y,   sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        frame_len = int(0.05 * sr)
        hop       = frame_len // 2
        frames    = librosa.util.frame(y, frame_length=frame_len, hop_length=hop).T
        energy_var = float(np.var(np.sum(frames ** 2, axis=1)))

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        mfcc_means = [float(np.mean(mfccs[i])) for i in range(5)]

        return {
            'audio_path': audio_path_str,
            'duration_sec': round(duration, 3),
            'rms': rms, 'peak': peak, 'par': par,
            'spectral_centroid': sc,
            'spectral_bandwidth': sbw,
            'spectral_rolloff': sro,
            'zero_crossing_rate': zcr,
            'energy_variance': energy_var,
            'mfcc_0': mfcc_means[0], 'mfcc_1': mfcc_means[1],
            'mfcc_2': mfcc_means[2], 'mfcc_3': mfcc_means[3],
            'mfcc_4': mfcc_means[4],
            'success': True,
        }
    except Exception as e:
        return {'audio_path': audio_path_str, 'success': False, 'error': str(e)}


# ============================================================
# HELPERS
# ============================================================
_TS_PATTERNS = [
    r'(\d{4})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})',
    r'(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})',
]

def _parse_timestamp(filename: str):
    name = Path(filename).stem
    for pat in _TS_PATTERNS:
        m = re.search(pat, name)
        if m:
            g = m.groups()
            try:
                return datetime(int(g[0]), int(g[1]), int(g[2]),
                                int(g[3]), int(g[4]), int(g[5]))
            except ValueError:
                continue
    return None


def _load_mech_data(folder_path: Path):
    """Load and normalise the first valid CSV found in folder_path."""
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
                # Strip unit suffix e.g. "0 mm"
                df[rain_col] = (df[rain_col].astype(str)
                                .str.replace('mm', '', regex=False)
                                .str.strip())
                df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce').fillna(0.0)
                return {'df': df, 'time_col': time_col,
                        'rain_col': rain_col, 'file': csv_file}
        except Exception as e:
            print(f"    [WARN] {csv_file.name}: {e}")
    return None


def _dur_category(d: float) -> str:
    if d < 5:   return '<5s'
    if d < 10:  return '5-10s'
    if d < 15:  return '10-15s'
    return '>15s'


# ============================================================
# CORE: process one pair
# ============================================================
def process_pair(audio_folder: Path, mech_folder: Path, dest_folder: Path) -> dict | None:
    print(f"\n{'='*68}")
    print(f"  Audio : {audio_folder.name}")
    print(f"  Mech  : {mech_folder.name}")
    print(f"{'='*68}")

    # 1. Load mechanical data
    mech = _load_mech_data(mech_folder)
    if mech is None:
        print("  SKIP: no mechanical data")
        return None
    mech_df  = mech['df']
    time_col = mech['time_col']
    rain_col = mech['rain_col']
    print(f"  Mech  : {len(mech_df)} rows | {time_col}, {rain_col}")

    # 2. Collect WAV paths
    audio_paths = sorted(audio_folder.rglob("*.wav"))
    n = len(audio_paths)
    print(f"  Audio : {n} files")
    if n == 0:
        print("  SKIP: no WAV files")
        return None

    # 3. Alignment in main process (fast: pure pandas merge_asof)
    print("  Building alignment table...")
    audio_data = []
    ts_fail = 0
    for ap in audio_paths:
        ts = _parse_timestamp(ap.name)
        if ts is None:
            ts_fail += 1
            continue
        audio_data.append({'audio_path': str(ap), 'timestamp': ts})

    if ts_fail:
        print(f"  [WARN] {ts_fail} files had unparseable timestamps — skipped")

    if not audio_data:
        print("  SKIP: no valid audio timestamps")
        return None

    audio_df = pd.DataFrame(audio_data).sort_values('timestamp')

    # merge_asof requires sorted inputs
    aligned_df = pd.merge_asof(
        audio_df,
        mech_df,
        left_on='timestamp',
        right_on=time_col,
        direction='nearest',
        tolerance=pd.Timedelta(minutes=2)
    )

    # Fill NaNs for values outside tolerance
    aligned_df['rainfall_mm'] = aligned_df[rain_col].fillna(0.0)
    aligned_df['is_aligned']  = aligned_df[time_col].notna()

    # Convert to dictionary for quick lookup by paths
    alignment = aligned_df.set_index('audio_path')[['timestamp', 'rainfall_mm', 'is_aligned']].to_dict('index')
    valid_paths = list(alignment.keys())
    print(f"  Aligned {len(valid_paths)}/{n} files to mech data")

    # 4. Parallel feature extraction
    print(f"  Extracting features with {NUM_WORKERS} workers ...")
    dest_folder.mkdir(parents=True, exist_ok=True)

    records = []
    dur_dist = {'<5s': 0, '5-10s': 0, '10-15s': 0, '>15s': 0}
    errors   = 0

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.imap_unordered(_extract_features, valid_paths, chunksize=100)
        with tqdm(total=len(valid_paths), desc="  Features", unit="file") as pbar:
            for result in results:
                pbar.update(1)
                if not result.get('success', False):
                    errors += 1
                    continue
                path_str = result['audio_path']
                align    = alignment[path_str]
                dur_cat  = _dur_category(result['duration_sec'])
                dur_dist[dur_cat] += 1
                records.append({
                    'audio_filename':  Path(path_str).name,
                    'audio_full_path': path_str,
                    'timestamp':       align['timestamp'],
                    'rainfall_mm':     align['rainfall_mm'],
                    'is_aligned':      align['is_aligned'],
                    'duration_category': dur_cat,
                    **{k: v for k, v in result.items()
                       if k not in ('audio_path', 'success', 'error')},
                })

    if not records:
        print("  WARN: no records produced")
        return None

    # 5. Save outputs
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df.to_csv(dest_folder / "cleaned_aligned_data.csv", index=False)

    metadata = {
        'audio_folder':      audio_folder.name,
        'mech_folder':       mech_folder.name,
        'total_audio_files': n,
        'processed_records': len(df),
        'aligned_records':   int(df['is_aligned'].sum()),
        'feature_errors':    errors,
        'duration_distribution': dur_dist,
        'rainy_samples':     int((df['rainfall_mm'] > 0).sum()),
        'dry_samples':       int((df['rainfall_mm'] == 0).sum()),
        'time_range': {
            'start': str(df['timestamp'].min()),
            'end':   str(df['timestamp'].max()),
        },
        'mech_data_file': str(mech['file']),
    }
    with open(dest_folder / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  DONE:")
    print(f"    Records    : {len(df):,}")
    print(f"    Aligned    : {metadata['aligned_records']:,}")
    print(f"    Rainy/Dry  : {metadata['rainy_samples']:,} / {metadata['dry_samples']:,}")
    print(f"    Errors     : {errors}")
    print(f"    Duration   : {dur_dist}")
    print(f"    Saved to   : {dest_folder / 'cleaned_aligned_data.csv'}")
    return metadata


# ============================================================
# MAIN
# ============================================================
def main():
    DESTINATION_DRIVE.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print(f"\n{'='*68}")
    print(f"  ACOUSTIC RAINFALL DATA CLEANING  —  {start_time:%Y-%m-%d %H:%M:%S}")
    print(f"{'='*68}")
    print(f"  Source      : {SOURCE_DRIVE}")
    print(f"  Destination : {DESTINATION_DRIVE}")
    print(f"  Workers     : {NUM_WORKERS}")
    print(f"  Pairs       : {len(FOLDER_PAIRS)}")
    print(f"  Delete src  : {DELETE_PROCESSED_FOLDERS}")
    print(f"  Resume mode : ON  (completed folders auto-skipped)")

    all_meta     = []
    success_count = 0

    for i, (audio_name, mech_name) in enumerate(FOLDER_PAIRS):
        audio_folder = SOURCE_DRIVE / audio_name
        mech_folder  = SOURCE_DRIVE / mech_name
        dest_folder  = DESTINATION_DRIVE / audio_name

        print(f"\n[{i+1}/{len(FOLDER_PAIRS)}]")

        if not audio_folder.exists():
            print(f"  SKIP: audio folder missing — {audio_name}")
            continue
        if not mech_folder.exists():
            print(f"  SKIP: mech folder missing — {mech_name}")
            continue

        # ── RESUME: skip if this pair already completed successfully ──
        done_marker = dest_folder / "metadata.json"
        done_csv    = dest_folder / "cleaned_aligned_data.csv"
        if done_marker.exists() and done_csv.exists():
            try:
                with open(done_marker) as f:
                    prior_meta = json.load(f)
                all_meta.append(prior_meta)
                success_count += 1
                print(f"  RESUME SKIP: already done — "
                      f"{prior_meta['processed_records']:,} records, "
                      f"rainy={prior_meta['rainy_samples']:,}")
                continue
            except Exception:
                pass  # corrupted metadata — reprocess

        try:
            meta = process_pair(audio_folder, mech_folder, dest_folder)
            if meta:
                success_count += 1
                all_meta.append(meta)

                if DELETE_PROCESSED_FOLDERS and audio_folder.resolve() != mech_folder.resolve():
                    try:
                        shutil.rmtree(audio_folder)
                        print(f"  Deleted: {audio_folder.name}")
                    except Exception as e:
                        print(f"  Delete failed: {e}")
            else:
                print(f"  SKIPPED: {audio_name}")

        except Exception as e:
            print(f"  ERROR [{audio_name}]: {e}")
            continue

    # Global summary
    elapsed = datetime.now() - start_time
    summary = {
        'run_date':              start_time.isoformat(),
        'elapsed_seconds':       elapsed.total_seconds(),
        'total_pairs':           len(FOLDER_PAIRS),
        'successfully_processed': success_count,
        'total_records':         sum(m['processed_records'] for m in all_meta),
        'total_aligned':         sum(m['aligned_records']   for m in all_meta),
        'total_rainy':           sum(m['rainy_samples']     for m in all_meta),
        'total_dry':             sum(m['dry_samples']       for m in all_meta),
        'total_errors':          sum(m['feature_errors']    for m in all_meta),
        'output_location':       str(DESTINATION_DRIVE),
        'per_folder':            all_meta,
    }
    with open(DESTINATION_DRIVE / "processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*68}")
    print(f"  COMPLETE  —  elapsed: {elapsed}")
    print(f"{'='*68}")
    print(f"  Pairs processed : {success_count}/{len(FOLDER_PAIRS)}")
    print(f"  Total records   : {summary['total_records']:,}")
    print(f"  Total aligned   : {summary['total_aligned']:,}")
    print(f"  Rainy samples   : {summary['total_rainy']:,}")
    print(f"  Dry samples     : {summary['total_dry']:,}")
    print(f"  Feature errors  : {summary['total_errors']:,}")
    print(f"  Summary JSON    : {DESTINATION_DRIVE / 'processing_summary.json'}")


if __name__ == "__main__":
    main()
