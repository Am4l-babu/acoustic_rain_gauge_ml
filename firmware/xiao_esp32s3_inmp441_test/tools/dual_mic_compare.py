"""
Host-side receiver + analyzer for MODE_DUAL_COMPARE (see src/main.cpp).

Sensor-migration validation tool: the project is switching its acoustic
sensor from a cheap analog condenser/comparator module (the one currently
used to build the training dataset) to the INMP441 I2S digital mic. Before
retiring the old sensor, this script captures both simultaneously so you
can see whether the trained ensemble's features shift meaningfully between
the two.

The firmware streams an interleaved stereo PCM16 frame at 8000 Hz:
  left channel  = INMP441 (digital I2S mic)
  right channel = analog condenser module (via ESP32 ADC)

This script:
  1. Accepts the TCP connection and records --seconds of stereo audio.
  2. Saves it as a stereo WAV (playable/inspectable in any audio tool).
  3. Splits it into two temporary mono WAVs and runs the *same* feature
     extraction the training pipeline uses (data_cleaning._extract_features)
     on each channel -- no reimplementation of the feature math.
  4. Prints a side-by-side comparison table.
  5. Saves a waveform + spectrum comparison plot as a PNG.

Usage:
    python dual_mic_compare.py --port 9494 --seconds 10 --out dual_capture.wav

Point wifi_config.h's SERVER_HOST/SERVER_PORT at the machine running this
script, set CAPTURE_MODE to MODE_DUAL_COMPARE in src/main.cpp, and re-upload.
"""

import argparse
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_cleaning import _extract_features  # noqa: E402  (path must be set up first)

SAMPLE_RATE = 8000   # must match SAMPLE_RATE in firmware/.../src/main.cpp
BYTES_PER_FRAME = 4   # 2 channels x int16 (2 bytes each)

FEATURES_TO_COMPARE = [
    "rms", "peak", "par",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "zero_crossing_rate", "energy_variance",
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
]


def record(port: int, seconds: float) -> np.ndarray:
    """Returns an (N, 2) float32 array in [-1, 1]: column 0 = INMP441, column 1 = analog module."""
    n_bytes = int(seconds * SAMPLE_RATE) * BYTES_PER_FRAME

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"Listening on 0.0.0.0:{port} -- waiting for the XIAO ESP32-S3 ...")
    conn, addr = srv.accept()
    print(f"Connection from {addr}. Recording {seconds:.1f}s ({n_bytes} bytes) ...")

    data = bytearray()
    try:
        while len(data) < n_bytes:
            chunk = conn.recv(65536)
            if not chunk:
                print("WARNING: connection closed early, using what was captured.")
                break
            data.extend(chunk)
    finally:
        conn.close()
        srv.close()

    if len(data) % BYTES_PER_FRAME:
        data = data[: len(data) - (len(data) % BYTES_PER_FRAME)]

    stereo_i16 = np.frombuffer(bytes(data), dtype="<i2").reshape(-1, 2)
    return stereo_i16.astype("float32") / 32768.0


def compare(stereo: np.ndarray, out_wav: Path, out_plot: Path):
    sf.write(str(out_wav), stereo, SAMPLE_RATE, subtype="PCM_16")
    print(f"Saved stereo capture -> {out_wav}")

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, col in [("inmp441", 0), ("analog_module", 1)]:
            path = Path(tmp) / f"{name}.wav"
            sf.write(str(path), stereo[:, col], SAMPLE_RATE, subtype="PCM_16")
            raw = _extract_features(str(path))
            if not raw.get("success"):
                print(f"WARNING: feature extraction failed for {name}: {raw.get('error')}")
                continue
            results[name] = raw

    if len(results) == 2:
        _print_table(results)
    _plot(stereo, results, out_plot)


def _print_table(results: dict):
    print("\n" + "=" * 72)
    print(f"{'Feature':<22}{'INMP441 (I2S)':>18}{'Analog module':>18}{'Ratio':>14}")
    print("=" * 72)
    inmp, analog = results["inmp441"], results["analog_module"]
    for feat in FEATURES_TO_COMPARE:
        a, b = inmp.get(feat, float("nan")), analog.get(feat, float("nan"))
        ratio = (a / b) if b not in (0, float("nan")) else float("nan")
        print(f"{feat:<22}{a:>18.5f}{b:>18.5f}{ratio:>14.3f}")
    print("=" * 72)
    print("Ratio far from 1.0 (or very different signs/scale) means the two")
    print("sensors disagree substantially on that feature -- expect the model")
    print("to need recalibration or retraining on INMP441-captured audio if")
    print("many ratios are large here, especially spectral_centroid/MFCCs.\n")


def _plot(stereo: np.ndarray, results: dict, out_plot: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(stereo.shape[0]) / SAMPLE_RATE
    n = stereo.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    axes[0].plot(t, stereo[:, 0], label="INMP441 (I2S)", alpha=0.8, linewidth=0.6)
    axes[0].plot(t, stereo[:, 1], label="Analog module", alpha=0.6, linewidth=0.6)
    axes[0].set_title("Time-domain waveform (normalized)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    for col, label in [(0, "INMP441 (I2S)"), (1, "Analog module")]:
        spec = np.abs(np.fft.rfft(stereo[:, col] * np.hanning(n)))
        spec_db = 20 * np.log10(spec + 1e-9)
        axes[1].plot(freqs, spec_db, label=label, linewidth=0.8)
    axes[1].set_title("Magnitude spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_xlim(0, SAMPLE_RATE / 2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(str(out_plot), dpi=150)
    print(f"Saved comparison plot -> {out_plot}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9494)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="dual_capture.wav")
    ap.add_argument("--plot-out", type=str, default="dual_capture_comparison.png")
    args = ap.parse_args()

    stereo = record(args.port, args.seconds)
    compare(stereo, Path(args.out), Path(args.plot_out))


if __name__ == "__main__":
    main()
