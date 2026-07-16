"""
Stage 8 — Master Feature Store: 176 acoustic features per clip, computed in
parallel across all 19 recording campaigns and written to chunked Parquet.

Supersedes src/advanced_feature_extraction.py's 63 features (that script's
Teager/Wavelet/Spectral-Flux/Histogram-Packet/Entropy families are all
reimplemented here, alongside the original 10 Stage-1 scalar features, MFCC,
and dense mel-band statistics) — run this instead of, not alongside, the
older script.

Usage:
    python src/features/master_feature_extraction.py                 # full run, all clips
    python src/features/master_feature_extraction.py --limit 2000     # smoke test
"""
import argparse
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pywt
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# OS/mount agnostic: derived from this script's own location rather than a
# hardcoded drive letter or mount path, since the HDD holding the dataset can
# show up as any letter on Windows (F:\, E:\, ...) or any mount point on Linux
# (/media/user/E-HDD, /mnt/..., ...). Project layout is always
# <root>/acoustic_rain_gauge_ml/src/features/master_feature_extraction.py, so
# the root three levels up is wherever the HDD actually mounted, on either OS.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = _PROJECT_ROOT / "arg_cleaned_dataset"
OUTPUT_DIR = _PROJECT_ROOT / "master_feature_store"
AUDIO_ROOT = _PROJECT_ROOT / "arg_dataset_unzip"

# cpu_count()-2 overcommits memory when other apps have most of the RAM in
# use — measured firsthand on Windows: with only ~1.9GB free out of 16GB
# total, spawning 18 workers (each importing numpy/scipy/librosa, ~200-400MB
# apiece under Windows' spawn-per-worker reimport) crashed several workers
# with "DLL load failed... paging file is too small". 6 is a safer default
# cross-platform; override with --workers if you've confirmed more RAM is
# free (rule of thumb: ~0.5GB free per worker on Windows). Linux's fork-based
# multiprocessing shares already-imported modules via copy-on-write instead of
# reimporting per worker, so it's usually safe to raise --workers higher there
# — check free RAM with CHECK_SPECS.sh first.
NUM_WORKERS = min(6, max(1, mp.cpu_count() - 2))
CHUNK_SIZE = 5000

# Native rate, confirmed in Stage 7 (README "Honest Notes"): every clip in the
# 10-15s training subset is 8000 Hz mono. Loading at 16000 would force librosa
# to resample every clip — pure upsampling that fabricates no new information
# above the true 4000 Hz Nyquist limit, and measured ~22ms/clip slower for it.
TARGET_SR = 8000

PACKET_DURATION = 0.5  # seconds, for histogram-packet analysis
NUM_HIST_BINS = 10


# ==========================================
# 1. FEATURE EXTRACTION FUNCTIONS
# ==========================================

def extract_time_and_teager(y):
    """Time-domain + Teager Energy Operator — raw loudness/shape + transient-impact detector."""
    rms = np.sqrt(np.mean(np.square(y)))
    peak = np.max(np.abs(y))
    teo = y[1:-1] ** 2 - y[:-2] * y[2:]

    return {
        "td_rms": rms,
        "td_peak": peak,
        "td_crest_factor": peak / (rms + 1e-10),
        "td_zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
        "td_skewness": stats.skew(y),
        "td_kurtosis": stats.kurtosis(y),
        "td_energy": np.sum(np.square(y)),
        "teo_mean": np.mean(teo),
        "teo_std": np.std(teo),
        "teo_max": np.max(teo),
        "teo_peak_to_mean": np.max(teo) / (np.mean(teo) + 1e-10),
    }


def extract_spectral(y, sr):
    """Frequency-domain descriptors, all sharing one STFT magnitude spectrogram."""
    S = np.abs(librosa.stft(y))

    # spectral_contrast's default band edges (fmin=200, n_bands=6) reach past
    # Nyquist at 8kHz (200 * 2**7 = 25600 Hz > 4000 Hz) and raise a
    # ParameterError — n_bands=3 keeps every band edge within range.
    n_bands = 3 if sr <= 8000 else 6

    flux = np.sum(np.diff(S, axis=1) ** 2, axis=0)

    return {
        "fd_spectral_centroid": np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)),
        "fd_spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)),
        "fd_spectral_rolloff_85": np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)),
        "fd_spectral_rolloff_95": np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)),
        "fd_spectral_flatness": np.mean(librosa.feature.spectral_flatness(S=S)),
        "fd_spectral_contrast_mean": np.mean(librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=n_bands)),
        "fd_spectral_contrast_std": np.std(librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=n_bands)),
        "fd_spectral_flux_mean": np.mean(flux),
        "fd_spectral_flux_std": np.std(flux),
        "fd_spectral_flux_max": np.max(flux),
    }


def extract_mfcc(y, sr):
    """13 MFCCs (mean + std) — perceptual spectral 'texture'."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {}
    for i in range(13):
        features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i}_std"] = np.std(mfccs[i])
    return features


def extract_mel_bands(y, sr, n_mels=40):
    """Dense per-band mel-spectrogram statistics (mean + std of each of 40 bands)."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec)
    features = {}
    for i in range(n_mels):
        features[f"mel_band_{i}_mean"] = np.mean(mel_db[i])
        features[f"mel_band_{i}_std"] = np.std(mel_db[i])
    return features


def extract_wavelets(y, wavelet="db4", level=4):
    """Multi-scale wavelet decomposition — energy ratio, entropy, variance per level."""
    coeffs = pywt.wavedec(y, wavelet, level=level)
    total_energy = sum(np.sum(c ** 2) for c in coeffs) + 1e-10

    features = {}
    for i, c in enumerate(coeffs):
        energy_ratio = np.sum(c ** 2) / total_energy
        p = c ** 2 / total_energy
        features[f"wavelet_l{i}_energy_ratio"] = energy_ratio
        features[f"wavelet_l{i}_entropy"] = -np.sum(p * np.log(p + 1e-10))
        features[f"wavelet_l{i}_var"] = np.var(c)
    return features


def extract_packets(y, sr, packet_duration=PACKET_DURATION, num_bins=NUM_HIST_BINS):
    """Histogram-packet analysis — temporal rhythm/variability of RMS and ZCR across 0.5s windows."""
    samples_per_packet = int(packet_duration * sr)
    total_packets = len(y) // samples_per_packet
    if total_packets < 3:
        return {}

    rms_p, zcr_p = [], []
    for i in range(total_packets):
        p = y[i * samples_per_packet:(i + 1) * samples_per_packet]
        rms_p.append(np.sqrt(np.mean(np.square(p))))
        zcr_p.append(np.mean(librosa.feature.zero_crossing_rate(p)))
    rms_p, zcr_p = np.array(rms_p), np.array(zcr_p)

    features = {}
    r_hist, _ = np.histogram(rms_p, bins=num_bins, density=True)
    z_hist, _ = np.histogram(zcr_p, bins=num_bins, density=True)
    for i in range(num_bins):
        features[f"rms_hist_bin_{i}"] = r_hist[i]
        features[f"zcr_hist_bin_{i}"] = z_hist[i]

    features.update({
        "rms_packet_mean": np.mean(rms_p), "rms_packet_std": np.std(rms_p),
        "rms_packet_skew": float(pd.Series(rms_p).skew()), "rms_packet_kurtosis": float(pd.Series(rms_p).kurtosis()),
        "zcr_packet_mean": np.mean(zcr_p), "zcr_packet_std": np.std(zcr_p),
        "zcr_packet_skew": float(pd.Series(zcr_p).skew()), "zcr_packet_kurtosis": float(pd.Series(zcr_p).kurtosis()),
    })
    return features


def extract_rhythm(y, sr):
    """Onset/tempo descriptors. Rain has no real musical beat, so tempo often
    collapses to librosa's tempogram prior — kept because it's cheap (~5ms)
    and left for SHAP (feature_selection.py) to judge, not assumed useful."""
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=oenv, sr=sr)[0]
    return {
        "rhythm_tempo_bpm": tempo,
        "rhythm_onset_strength_mean": np.mean(oenv),
        "rhythm_onset_strength_std": np.std(oenv),
        "rhythm_attack_rate": np.mean(np.diff(oenv) > 0),
        "rhythm_decay_rate": np.mean(np.diff(oenv) < 0),
    }


def extract_all_features(args):
    file_path, metadata = args
    try:
        y, sr = librosa.load(str(file_path), sr=TARGET_SR, duration=None)
        if len(y) < TARGET_SR:
            return None

        features = {}
        features.update(extract_time_and_teager(y))
        features.update(extract_spectral(y, sr))
        features.update(extract_mfcc(y, sr))
        features.update(extract_mel_bands(y, sr))
        features.update(extract_wavelets(y))
        features.update(extract_packets(y, sr))
        features.update(extract_rhythm(y, sr))
        features.update(metadata)
        return features
    except Exception:
        return None


# ==========================================
# 2. MASTER PIPELINE
# ==========================================

def resolve_audio_path(raw_path, audio_root):
    """Remap audio_full_path's baked-in drive letter/mount point to wherever
    the dataset actually lives now, anchored on the "arg_dataset_unzip" folder
    name so a plugged-in HDD landing on a different letter (Windows) or mount
    point (Linux) doesn't break every lookup. audio_full_path was written on
    Windows (e.g. "F:\\arg_dataset_unzip\\..."), so splitting must handle
    backslashes explicitly — pathlib's PurePosixPath on Linux won't treat
    backslash as a separator and would return the whole string as one part."""
    normalized = str(raw_path).replace("\\", "/")
    parts = normalized.split("/")
    if "arg_dataset_unzip" in parts:
        idx = parts.index("arg_dataset_unzip")
        return audio_root.joinpath(*parts[idx + 1:])
    return Path(raw_path)


def build_store(limit=None, data_dir=None, output_dir=None, audio_root=None, workers=None):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    audio_root = Path(audio_root) if audio_root else AUDIO_ROOT
    num_workers = workers if workers else NUM_WORKERS
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("MASTER FEATURE STORE — 176 features/clip")
    print("=" * 70)
    print(f"Data directory   : {data_dir}")
    print(f"Audio root       : {audio_root}")
    print(f"Output directory : {output_dir}")
    print(f"Workers          : {num_workers}")
    print(f"Sample rate      : {TARGET_SR} Hz (native, no resample)")
    print("=" * 70)

    print("\nScanning dataset...")
    all_csvs = list(data_dir.glob("**/cleaned_aligned_data.csv"))
    tasks = []
    missing = 0
    checked = 0
    # With --limit set (smoke test), stop checking path.exists() as soon as we
    # have enough valid tasks — each check is a filesystem stat against a
    # mechanical HDD, and scanning all ~780k rows unconditionally before
    # slicing to `limit` made --limit 2000 take as long as the full run.
    scan_done = False
    for csv_file in all_csvs:
        if scan_done:
            break
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            checked += 1
            path = resolve_audio_path(row["audio_full_path"], audio_root)
            if path.exists():
                tasks.append((path, {
                    "timestamp": row["timestamp"],
                    "rainfall_mm": row["rainfall_mm"],
                    "is_rainy": 1 if row["rainfall_mm"] > 0 else 0,
                    "source_folder": csv_file.parent.name,
                    "audio_filename": row["audio_filename"],
                }))
            else:
                missing += 1

            if limit and len(tasks) >= limit:
                scan_done = True
                break

    if missing:
        total = missing + len(tasks)
        print(f"WARNING: {missing}/{total} audio files checked were not found under {audio_root} "
              f"({missing / total:.1%}) — check audio_root / HDD drive letter if this looks high.")

    print(f"Found {len(tasks)} files (checked {checked} rows). Starting extraction with {num_workers} workers...")

    if len(tasks) == 0:
        print("ABORTING: no valid audio files found — check --data-dir / --audio-root and that the HDD is connected.")
        return

    # Resume-safety: skip chunks whose output already exists (matches the
    # resume-on-restart convention used by data_cleaning.py for multi-hour runs).
    chunk_num = 0
    t_start = time.time()
    for i in range(0, len(tasks), CHUNK_SIZE):
        chunk_num += 1
        out_path = output_dir / f"master_chunk_{chunk_num:03d}.parquet"
        if out_path.exists():
            print(f"Chunk {chunk_num} already done, skipping ({out_path.name})")
            continue

        chunk = tasks[i:i + CHUNK_SIZE]
        print(f"\nChunk {chunk_num}/{(len(tasks) - 1) // CHUNK_SIZE + 1}...")

        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(extract_all_features, chunk, chunksize=32), total=len(chunk)))

        valid = [r for r in results if r is not None]
        if valid:
            pd.DataFrame(valid).to_parquet(out_path, engine="pyarrow", compression="snappy")
            print(f"Saved {len(valid)}/{len(chunk)} records -> {out_path.name}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"EXTRACTION COMPLETE — {len(tasks)} clips in {elapsed / 3600:.2f}h ({elapsed / max(len(tasks), 1) * 1000:.1f} ms/clip)")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N clips (smoke test / timing pilot)")
    parser.add_argument("--data-dir", type=str, default=None, help=f"Override dataset CSV directory (default: {DATA_DIR})")
    parser.add_argument("--output-dir", type=str, default=None, help=f"Override output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--audio-root", type=str, default=None, help=f"Override audio root, i.e. wherever arg_dataset_unzip lives (default: {AUDIO_ROOT})")
    parser.add_argument("--workers", type=int, default=None, help=f"Override parallel worker count (default: {NUM_WORKERS} — raise only if you've confirmed enough free RAM, ~0.5GB/worker)")
    args = parser.parse_args()
    build_store(limit=args.limit, data_dir=args.data_dir, output_dir=args.output_dir, audio_root=args.audio_root, workers=args.workers)
