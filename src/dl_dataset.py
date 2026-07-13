"""
Waveform Dataset + GPU-batched MFCC extraction
=================================================
Supports train_dl_model.py (the SARID-style deep-learning ablation).

Every clip referenced by train.csv/test.csv is a fixed 80,000 samples
(10.0s @ 8kHz, confirmed empirically across all 19 recording campaigns), so
no padding/truncation logic is needed for the common case.

IMPORTANT: source audio lives on F:\\arg_dataset_unzip, a mechanical HDD.
Measured directly: sequential reads are ~7.6ms/file, but reads in random
order (i.e. a shuffled DataLoader touching files scattered across the whole
dataset) are ~55ms/file -- a 7.3x penalty from seek time. A shuffled 40,000-
row epoch would cost ~37 minutes of I/O alone, repeated every epoch, which
is infeasible. `precompute_mfcc()` reads every clip exactly ONCE, in the
DataFrame's existing (mostly locally-sequential) order, and caches the
resulting MFCC tensor in memory -- training epochs then shuffle and index
that in-memory cache instead of re-reading WAVs from disk.
"""

import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

SAMPLE_RATE = 8000
CLIP_SAMPLES = 80_000  # 10.0s @ 8kHz

# Auto-detected: two levels up from src/ is the HDD root, whether that's a
# Windows drive letter or a Linux mount point -- same convention as
# master_feature_extraction.py's resolve_audio_path().
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_ROOT = _PROJECT_ROOT.parent / "arg_dataset_unzip"


def resolve_audio_path(raw_path, audio_root):
    """audio_full_path in train.csv/test.csv was baked in at Stage 1 time on
    whatever machine/drive-letter/mount-point was current then (e.g.
    "F:\\arg_dataset_unzip\\..." from Windows). Remap it onto wherever the
    dataset actually lives now by anchoring on the "arg_dataset_unzip" folder
    name, so this works whether the HDD is plugged into Windows (drive
    letter) or Linux (mount point) -- must split on backslash explicitly
    since PurePosixPath on Linux won't treat it as a separator."""
    normalized = str(raw_path).replace("\\", "/")
    parts = normalized.split("/")
    if "arg_dataset_unzip" in parts:
        idx = parts.index("arg_dataset_unzip")
        return str(audio_root.joinpath(*parts[idx + 1:]))
    return str(raw_path)


class WaveformDataset(Dataset):
    """Returns (raw waveform, rainfall_mm) pairs. No feature extraction here --
    that happens batched on GPU in the training loop via MFCCExtractor."""

    def __init__(self, df, audio_root=None):
        audio_root = Path(audio_root) if audio_root else DEFAULT_AUDIO_ROOT
        self.paths = [resolve_audio_path(p, audio_root) for p in df["audio_full_path"]]
        self.targets = df["rainfall_mm"].to_numpy(dtype="float32")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        y, _sr = sf.read(self.paths[idx], dtype="float32")
        if len(y) != CLIP_SAMPLES:
            # Defensive: a handful of clips may be off by a few samples due to
            # encoder rounding. Pad/truncate rather than let one bad clip crash a batch.
            if len(y) < CLIP_SAMPLES:
                y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
            else:
                y = y[:CLIP_SAMPLES]
        return torch.from_numpy(y), self.targets[idx]


class MFCCExtractor(torch.nn.Module):
    """GPU-batched MFCC, same pattern as data_cleaning_gpu.py's GPUFeatureExtractor.

    Returns RAW (un-normalized) MFCC. An earlier version divided each clip by
    its own max abs value (matching SARID's `librosa.util.normalize`), but
    that erases absolute loudness/energy -- exactly the signal the SHAP
    ranking on the scalar-feature pipeline identified as the strongest
    rainfall_mm predictor (mel_band_8_mean, td_peak, td_rms, td_energy all
    rank in the top 15 of 175 features, and none of those are per-clip
    normalized). Harder rain hits louder; per-clip peak-normalizing throws
    that away before the model ever sees it. Global normalization (fixed
    train-set-derived mean/std, applied identically to every clip) is done
    once in train_dl_model.py after precompute instead, so relative loudness
    between clips survives.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mfcc: int = 40,
                 n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
        super().__init__()
        self.mfcc = T.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels},
        )

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """waveforms: (B, T_samples) -> (B, n_mfcc, T_frames), raw scale."""
        return self.mfcc(waveforms)


def precompute_mfcc(df: pd.DataFrame, extractor: "MFCCExtractor", device,
                     batch_size: int = 256, num_workers: int = 0, audio_root=None,
                     checkpoint_dir=None):
    """
    Reads every clip in `df` exactly once, in DataFrame order (fast: measured
    ~7.6ms/file sequential vs ~55ms/file random on the source HDD), and
    returns (mfcc_tensor, target_tensor) cached on CPU. shuffle=False here is
    deliberate -- shuffling belongs to the downstream TensorDataset built from
    the cached result, not to this one-time disk read.

    Note: num_workers defaults to 0 (no multiprocessing) to avoid Windows spawn
    issues. On Linux, can be increased to 8+ for faster I/O.

    If `checkpoint_dir` is given, each batch's MFCC is written to its own
    small file (`batch_00000.pt`, ...) as soon as it's computed. On a fresh
    call, any batch files already present are detected and skipped -- so an
    interruption (crash, unplugged drive) loses at most one in-flight batch
    (~256 clips, a few seconds) instead of the whole multi-hour precompute.
    The per-batch files are merged into the returned tensors and the
    directory is deleted once the split completes.
    """
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    start_batch = 0
    if checkpoint_dir and checkpoint_dir.exists():
        existing = sorted(checkpoint_dir.glob("batch_*.pt"))
        start_batch = len(existing)
        if start_batch:
            print(f"    Found {start_batch} cached batches in {checkpoint_dir} "
                  f"({start_batch * batch_size:,} clips) -- resuming from there...", flush=True)

    dataset = WaveformDataset(df, audio_root=audio_root)
    start_idx = start_batch * batch_size
    remaining = Subset(dataset, range(start_idx, len(dataset))) if start_idx else dataset

    print(f"    Creating DataLoader (num_workers={num_workers})...", flush=True)
    loader = DataLoader(remaining, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    print(f"    DataLoader ready, iterating {len(remaining):,} remaining clips "
          f"(of {len(df):,} total) in batches of {batch_size}...", flush=True)

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    in_memory_mfcc, in_memory_targets = [], []
    for i, (waveforms, y) in enumerate(loader):
        batch_idx = start_batch + i
        waveforms = waveforms.to(device)
        mfcc = extractor(waveforms).cpu()
        if checkpoint_dir:
            torch.save({"mfcc": mfcc, "y": y}, checkpoint_dir / f"batch_{batch_idx:05d}.pt")
        else:
            in_memory_mfcc.append(mfcc)
            in_memory_targets.append(y)
        if (i + 1) % 10 == 0:
            n_done = start_idx + (i + 1) * batch_size
            pct = 100 * n_done / len(df)
            print(f"      Batch {batch_idx + 1}: {n_done:,} / {len(df):,} clips ({pct:.0f}%)", flush=True)

    print(f"    Concatenating cached tensors...", flush=True)
    if not checkpoint_dir:
        return torch.cat(in_memory_mfcc), torch.cat(in_memory_targets)

    all_mfcc, all_targets = [], []
    for f in sorted(checkpoint_dir.glob("batch_*.pt")):
        chunk = torch.load(f)
        all_mfcc.append(chunk["mfcc"])
        all_targets.append(chunk["y"])
    result = torch.cat(all_mfcc), torch.cat(all_targets)
    shutil.rmtree(checkpoint_dir)
    return result
