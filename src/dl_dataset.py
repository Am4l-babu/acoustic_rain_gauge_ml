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

import pandas as pd
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 8000
CLIP_SAMPLES = 80_000  # 10.0s @ 8kHz


class WaveformDataset(Dataset):
    """Returns (raw waveform, rainfall_mm) pairs. No feature extraction here --
    that happens batched on GPU in the training loop via MFCCExtractor."""

    def __init__(self, df):
        self.paths = df["audio_full_path"].tolist()
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

    Per-clip peak normalization (divide by max abs value) matches SARID's own
    `librosa.util.normalize` step in data_processing.py.
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
        """waveforms: (B, T_samples) -> (B, n_mfcc, T_frames), normalized to [-1, 1]."""
        mfcc = self.mfcc(waveforms)
        peak = mfcc.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return mfcc / peak


def precompute_mfcc(df: pd.DataFrame, extractor: "MFCCExtractor", device,
                     batch_size: int = 256, num_workers: int = 0):
    """
    Reads every clip in `df` exactly once, in DataFrame order (fast: measured
    ~7.6ms/file sequential vs ~55ms/file random on the source HDD), and
    returns (mfcc_tensor, target_tensor) cached on CPU. shuffle=False here is
    deliberate -- shuffling belongs to the downstream TensorDataset built from
    the cached result, not to this one-time disk read.

    Note: num_workers defaults to 0 (no multiprocessing) to avoid Windows spawn
    issues. On Linux, can be increased to 8+ for faster I/O.
    """
    print(f"    Creating DataLoader (num_workers={num_workers})...", flush=True)
    loader = DataLoader(WaveformDataset(df), batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    print(f"    DataLoader ready, iterating {len(df):,} clips in batches of {batch_size}...", flush=True)

    all_mfcc, all_targets = [], []
    for i, (waveforms, y) in enumerate(loader):
        waveforms = waveforms.to(device)
        all_mfcc.append(extractor(waveforms).cpu())
        all_targets.append(y)
        if (i + 1) % 10 == 0:
            n_done = (i + 1) * batch_size
            pct = 100 * n_done / len(df)
            print(f"      Batch {i+1}: {n_done:,} / {len(df):,} clips ({pct:.0f}%)", flush=True)

    print(f"    Concatenating cached tensors...", flush=True)
    return torch.cat(all_mfcc), torch.cat(all_targets)
