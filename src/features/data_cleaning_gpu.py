"""
Acoustic Rainfall Dataset Cleaning — GPU-Accelerated Version
=============================================================
Uses torchaudio + CUDA for batched feature extraction on the RTX 4060.

Architecture
------------
  I/O threads (CPU)  : load WAV files in parallel using ThreadPoolExecutor
  GPU batch          : stack audio tensors, run all transforms in one pass
  Main process       : alignment, result collection, CSV writing

Expected speedup vs CPU version: 4-8x
"""

import os, shutil, json, re, warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
SOURCE_DRIVE      = Path(r"F:\arg_dataset_unzip")
DESTINATION_DRIVE = Path(r"D:\arg_cleaned_dataset")
DELETE_PROCESSED_FOLDERS = False

SAMPLE_RATE   = 16000
BATCH_SIZE    = 64      # files per GPU batch (tune to VRAM; 8GB -> 64-128)
IO_WORKERS    = 8       # threads for parallel file loading

DESTINATION_DRIVE.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# FOLDER PAIRS
# ============================================================
FOLDER_PAIRS = [
    ("April_2024_rainfall_audios_with_mech_data",       "April_2024_rainfall_audios_with_mech_data"),
    ("December_2023_rainfall_audios_with_mech_data",    "December_2023_rainfall_audios_with_mech_data"),
    ("January_2024_rainfall_audios_with_mech_data",     "January_2024_rainfall_audios_with_mech_data"),
    ("July_2024_rainfall_audios_with_mech_data",        "July_2024_rainfall_audios_with_mech_data"),
    ("May_2024_rainfall_audios_with_mechanical_data",   "May_2024_rainfall_audios_with_mechanical_data"),
    ("November_2023_rainfall_audios_with_mech_data",    "November_2023_rainfall_audios_with_mech_data"),
    ("September_2024_rainfall_audios_with_mech_data",   "September_2024_rainfall_audios_with_mech_data"),
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
# GPU TRANSFORMS  (built once, reused for all batches)
# ============================================================
class GPUFeatureExtractor(torch.nn.Module):
    def __init__(self, sr=SAMPLE_RATE, n_fft=512, hop_length=256, n_mfcc=5):
        super().__init__()
        self.sr          = sr
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_mfcc      = n_mfcc
        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length,
                                          power=2.0).to(DEVICE)
        self.mfcc        = T.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
                                   melkwargs={"n_fft": n_fft,
                                              "hop_length": hop_length,
                                              "n_mels": 64}).to(DEVICE)

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor) -> dict:
        """
        waveforms: (B, T) float32 on DEVICE
        Returns dict of numpy arrays, one value per sample.
        """
        B = waveforms.shape[0]

        # Basic amplitude stats
        rms  = waveforms.pow(2).mean(dim=1).sqrt()          # (B,)
        peak = waveforms.abs().amax(dim=1)                  # (B,)
        par  = torch.where(rms > 0, peak / rms,
                           torch.zeros(B, device=DEVICE))   # (B,)

        # Zero-crossing rate
        signs = torch.sign(waveforms)
        zcr   = (signs[:, 1:] != signs[:, :-1]).float().mean(dim=1)  # (B,)

        # Energy variance over 50ms frames
        frame_len = int(0.05 * self.sr)
        hop       = frame_len // 2
        T_len     = waveforms.shape[1]
        n_frames  = (T_len - frame_len) // hop + 1
        frames    = waveforms.unfold(1, frame_len, hop)          # (B, n_frames, frame_len)
        frame_energy = frames.pow(2).sum(dim=2)                  # (B, n_frames)
        energy_var   = frame_energy.var(dim=1)                   # (B,)

        # Power spectrogram  (B, F, T_spec)
        spec = self.spectrogram(waveforms)                       # (B, F, T_spec)
        freq_bins = spec.shape[1]
        freqs     = torch.linspace(0, self.sr / 2, freq_bins,
                                   device=DEVICE)                # (F,)

        spec_sum  = spec.sum(dim=2)                              # (B, F)  marginal over time
        total_pow = spec_sum.sum(dim=1, keepdim=True) + 1e-10   # (B, 1)

        # Spectral centroid
        sc  = (spec_sum * freqs.unsqueeze(0)).sum(dim=1) / total_pow.squeeze(1)

        # Spectral bandwidth
        diff = (freqs.unsqueeze(0) - sc.unsqueeze(1)).pow(2)
        sbw  = ((spec_sum * diff).sum(dim=1) / total_pow.squeeze(1)).sqrt()

        # Spectral rolloff (85% of energy)
        cumsum = spec_sum.cumsum(dim=1) / total_pow
        rolloff_idx = (cumsum < 0.85).sum(dim=1).clamp(0, freq_bins - 1)
        sro   = freqs[rolloff_idx]

        # MFCCs  (B, n_mfcc, T_mfcc)
        mfccs = self.mfcc(waveforms)                             # (B, n_mfcc, T_mfcc)
        mfcc_means = mfccs.mean(dim=2)                           # (B, n_mfcc)

        return {
            "rms":                rms.cpu().numpy(),
            "peak":               peak.cpu().numpy(),
            "par":                par.cpu().numpy(),
            "zero_crossing_rate": zcr.cpu().numpy(),
            "energy_variance":    energy_var.cpu().numpy(),
            "spectral_centroid":  sc.cpu().numpy(),
            "spectral_bandwidth": sbw.cpu().numpy(),
            "spectral_rolloff":   sro.cpu().numpy(),
            **{f"mfcc_{i}": mfcc_means[:, i].cpu().numpy()
               for i in range(self.n_mfcc)},
        }


# ============================================================
# HELPERS
# ============================================================
_TS_PATTERNS = [
    r"(\d{4})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})",
    r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})",
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
    for csv_file in folder_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip().str.lower()
            time_col = rain_col = None
            for col in df.columns:
                if ("time" in col or "timestamp" in col) and time_col is None:
                    time_col = col
                if col in ("rainfall (mm)", "rainfall") and rain_col is None:
                    rain_col = col
            if rain_col is None:
                for col in df.columns:
                    if "payload" in col or "measurement" in col:
                        rain_col = col; break
            if time_col and rain_col:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
                df[rain_col] = (df[rain_col].astype(str)
                                .str.replace("mm", "", regex=False).str.strip())
                df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0)
                return {"df": df, "time_col": time_col,
                        "rain_col": rain_col, "file": csv_file}
        except Exception as e:
            print(f"    [WARN] {csv_file.name}: {e}")
    return None


def _dur_category(d: float) -> str:
    if d < 5:   return "<5s"
    if d < 10:  return "5-10s"
    if d < 15:  return "10-15s"
    return ">15s"


def _load_audio(path_str: str, target_sr: int = SAMPLE_RATE):
    """Load one WAV file and return (mono float32 tensor, sample_rate, path)."""
    try:
        wav, sr = torchaudio.load(path_str)          # (C, T), int16 or float
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)       # stereo -> mono
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.squeeze(0), target_sr, path_str, None
    except Exception as e:
        return None, None, path_str, str(e)


# ============================================================
# CORE
# ============================================================
def process_pair(audio_folder: Path, mech_folder: Path,
                 dest_folder: Path, extractor: GPUFeatureExtractor) -> dict | None:

    print(f"\n{'='*66}")
    print(f"  Audio : {audio_folder.name}")
    print(f"  Mech  : {mech_folder.name}")
    print(f"{'='*66}")

    mech = _load_mech_data(mech_folder)
    if mech is None:
        print("  SKIP: no mechanical data"); return None

    mech_df  = mech["df"]
    time_col = mech["time_col"]
    rain_col = mech["rain_col"]
    print(f"  Mech  : {len(mech_df)} rows | {time_col}, {rain_col}")

    audio_paths = sorted(audio_folder.rglob("*.wav"))
    n = len(audio_paths)
    print(f"  Audio : {n} files")
    if n == 0:
        print("  SKIP: no WAV files"); return None

    # Build alignment table in main process
    print("  Building alignment table...")
    alignment = {}
    ts_fail = 0
    for ap in audio_paths:
        ts = _parse_timestamp(ap.name)
        if ts is None:
            ts_fail += 1; continue
        diffs = (mech_df[time_col] - ts).abs()
        idx   = diffs.idxmin()
        if diffs.min() < timedelta(minutes=2):
            rain_val   = float(mech_df.loc[idx, rain_col])
            is_aligned = True
        else:
            rain_val   = 0.0
            is_aligned = False
        alignment[str(ap)] = {"timestamp": ts, "rainfall_mm": rain_val,
                               "is_aligned": is_aligned}
    if ts_fail:
        print(f"  [WARN] {ts_fail} files had unparseable timestamps")

    valid_paths = [str(ap) for ap in audio_paths if str(ap) in alignment]
    print(f"  Aligned {len(valid_paths)}/{n} files")

    dest_folder.mkdir(parents=True, exist_ok=True)

    records  = []
    dur_dist = {"<5s": 0, "5-10s": 0, "10-15s": 0, ">15s": 0}
    errors   = 0

    print(f"  GPU batch processing (batch={BATCH_SIZE}, io_workers={IO_WORKERS}) ...")

    with tqdm(total=len(valid_paths), desc="  Features", unit="file") as pbar:
        # Process in chunks
        for chunk_start in range(0, len(valid_paths), BATCH_SIZE):
            batch_paths = valid_paths[chunk_start: chunk_start + BATCH_SIZE]

            # Load audio files in parallel I/O threads
            with ThreadPoolExecutor(max_workers=IO_WORKERS) as io_pool:
                loaded = list(io_pool.map(_load_audio, batch_paths))

            # Separate successes and failures
            ok_waveforms, ok_paths, ok_durations = [], [], []
            for wav, sr, path_str, err in loaded:
                if wav is None:
                    errors += 1
                else:
                    ok_waveforms.append(wav)
                    ok_paths.append(path_str)
                    ok_durations.append(len(wav) / SAMPLE_RATE)

            if not ok_waveforms:
                pbar.update(len(batch_paths))
                continue

            # Pad to same length and stack -> GPU
            max_len   = max(w.shape[0] for w in ok_waveforms)
            padded    = torch.zeros(len(ok_waveforms), max_len)
            for i, w in enumerate(ok_waveforms):
                padded[i, :w.shape[0]] = w
            batch_gpu = padded.to(DEVICE)

            # Extract all features in one GPU pass
            feats = extractor(batch_gpu)
            del batch_gpu, padded          # free VRAM immediately

            # Collect records
            for i, path_str in enumerate(ok_paths):
                dur     = ok_durations[i]
                dur_cat = _dur_category(dur)
                dur_dist[dur_cat] += 1
                align   = alignment[path_str]
                records.append({
                    "audio_filename":    Path(path_str).name,
                    "audio_full_path":   path_str,
                    "timestamp":         align["timestamp"],
                    "rainfall_mm":       align["rainfall_mm"],
                    "is_aligned":        align["is_aligned"],
                    "duration_sec":      round(dur, 3),
                    "duration_category": dur_cat,
                    **{k: float(v[i]) for k, v in feats.items()},
                })

            pbar.update(len(batch_paths))

    if not records:
        print("  WARN: no records produced"); return None

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(dest_folder / "cleaned_aligned_data.csv", index=False)

    metadata = {
        "audio_folder":      audio_folder.name,
        "mech_folder":       mech_folder.name,
        "total_audio_files": n,
        "processed_records": len(df),
        "aligned_records":   int(df["is_aligned"].sum()),
        "feature_errors":    errors,
        "duration_distribution": dur_dist,
        "rainy_samples":     int((df["rainfall_mm"] > 0).sum()),
        "dry_samples":       int((df["rainfall_mm"] == 0).sum()),
        "time_range": {
            "start": str(df["timestamp"].min()),
            "end":   str(df["timestamp"].max()),
        },
        "mech_data_file": str(mech["file"]),
    }
    with open(dest_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  DONE:")
    print(f"    Records    : {len(df):,}")
    print(f"    Aligned    : {metadata['aligned_records']:,}")
    print(f"    Rainy/Dry  : {metadata['rainy_samples']:,} / {metadata['dry_samples']:,}")
    print(f"    Errors     : {errors}")
    print(f"    Duration   : {dur_dist}")
    return metadata


def main():
    start_time = datetime.now()
    print(f"\n{'='*66}")
    print(f"  ACOUSTIC RAINFALL CLEANING (GPU)  —  {start_time:%Y-%m-%d %H:%M:%S}")
    print(f"{'='*66}")
    print(f"  Source      : {SOURCE_DRIVE}")
    print(f"  Destination : {DESTINATION_DRIVE}")
    print(f"  Device      : {DEVICE}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  I/O workers : {IO_WORKERS}")
    print(f"  Pairs       : {len(FOLDER_PAIRS)}")
    print(f"  Resume mode : ON  (completed folders auto-skipped)")

    extractor    = GPUFeatureExtractor().to(DEVICE)
    all_meta     = []
    success_count = 0

    for i, (audio_name, mech_name) in enumerate(FOLDER_PAIRS):
        audio_folder = SOURCE_DRIVE / audio_name
        mech_folder  = SOURCE_DRIVE / mech_name
        dest_folder  = DESTINATION_DRIVE / audio_name

        print(f"\n[{i+1}/{len(FOLDER_PAIRS)}]")

        if not audio_folder.exists():
            print(f"  SKIP: audio folder missing — {audio_name}"); continue
        if not mech_folder.exists():
            print(f"  SKIP: mech folder missing — {mech_name}"); continue

        # Resume: skip if already completed
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
                pass

        try:
            meta = process_pair(audio_folder, mech_folder, dest_folder, extractor)
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

    elapsed = datetime.now() - start_time
    summary = {
        "run_date":               start_time.isoformat(),
        "elapsed_seconds":        elapsed.total_seconds(),
        "device":                 str(DEVICE),
        "total_pairs":            len(FOLDER_PAIRS),
        "successfully_processed": success_count,
        "total_records":          sum(m["processed_records"] for m in all_meta),
        "total_aligned":          sum(m["aligned_records"]   for m in all_meta),
        "total_rainy":            sum(m["rainy_samples"]     for m in all_meta),
        "total_dry":              sum(m["dry_samples"]       for m in all_meta),
        "total_errors":           sum(m["feature_errors"]    for m in all_meta),
        "output_location":        str(DESTINATION_DRIVE),
        "per_folder":             all_meta,
    }
    with open(DESTINATION_DRIVE / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*66}")
    print(f"  COMPLETE  —  elapsed: {elapsed}")
    print(f"{'='*66}")
    print(f"  Pairs processed : {success_count}/{len(FOLDER_PAIRS)}")
    print(f"  Total records   : {summary['total_records']:,}")
    print(f"  Total aligned   : {summary['total_aligned']:,}")
    print(f"  Rainy samples   : {summary['total_rainy']:,}")
    print(f"  Dry samples     : {summary['total_dry']:,}")
    print(f"  Feature errors  : {summary['total_errors']:,}")
    print(f"  Summary JSON    : {DESTINATION_DRIVE / 'processing_summary.json'}")


if __name__ == "__main__":
    main()
