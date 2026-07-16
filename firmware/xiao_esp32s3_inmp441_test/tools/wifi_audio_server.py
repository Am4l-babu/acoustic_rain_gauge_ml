"""
Host-side receiver for MODE_WIFI_STREAM (see src/main.cpp).

The XIAO ESP32-S3 streams continuous raw 16-bit mono PCM @ 8000 Hz over a
TCP socket. This script accepts that connection, buffers the stream into
fixed-length windows, saves each window as a WAV file, and feeds it
straight into the repo's existing trained ensemble
(src/predict.py: predict_ensemble) -- no separate on-device model, no
feature-extraction reimplementation.

Window length defaults to 12.0s, which sits inside the "10-15s" duration
bucket predict.py expects (see TRAINED_DURATION_CATEGORY in src/predict.py
and CLIP_SAMPLES in src/dl_dataset.py) -- anything outside that bucket is
flagged out-of-distribution by the model's own reporting, not silently
ignored.

Usage:
    python wifi_audio_server.py --port 9494
    python wifi_audio_server.py --port 9494 --window 12 --save-clips captures/

Then set SERVER_HOST to this machine's LAN IP and SERVER_PORT to match
in firmware/xiao_esp32s3_inmp441_test/include/wifi_config.h before
building/uploading the firmware.
"""

import argparse
import socket
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from predict import predict_ensemble  # noqa: E402  (path must be set up first)

SAMPLE_RATE = 8000   # must match SAMPLE_RATE in firmware/.../src/main.cpp
BYTES_PER_SAMPLE = 2  # int16


def handle_connection(conn: socket.socket, window_sec: float, save_dir: Path | None):
    window_bytes = int(window_sec * SAMPLE_RATE) * BYTES_PER_SAMPLE
    buf = bytearray()

    print(f"Client connected. Buffering {window_sec:.1f}s windows "
          f"({window_bytes} bytes) at {SAMPLE_RATE} Hz.")

    while True:
        chunk = conn.recv(65536)
        if not chunk:
            print("Client disconnected.")
            return
        buf.extend(chunk)

        while len(buf) >= window_bytes:
            window = bytes(buf[:window_bytes])
            del buf[:window_bytes]
            _process_window(window, save_dir)


def _process_window(pcm_bytes: bytes, save_dir: Path | None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        wav_path = save_dir / f"clip_{ts}.wav"
    else:
        wav_path = Path(tempfile.gettempdir()) / f"rain_clip_{ts}.wav"

    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype("float32") / 32768.0
    sf.write(str(wav_path), samples, SAMPLE_RATE, subtype="PCM_16")

    try:
        result = predict_ensemble(str(wav_path))
    except Exception as exc:
        print(f"[{ts}] ERROR scoring clip: {exc}")
        return

    flag = "  (OUT-OF-DISTRIBUTION duration!)" if result["out_of_distribution"] else ""
    verdict = "RAINY" if result["is_rainy"] else "dry"
    print(f"[{ts}] {result['duration_sec']:.1f}s  "
          f"P(rain)={result['rain_probability']:.3f}  "
          f"est={result['estimated_rainfall_mm']:.3f}mm  -> {verdict}{flag}")

    if save_dir is None:
        wav_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9494)
    ap.add_argument("--window", type=float, default=12.0,
                     help="Window length in seconds (keep within 10-15s to match training)")
    ap.add_argument("--save-clips", type=str, default=None,
                     help="Directory to permanently save every scored clip "
                          "(useful for building a field dataset). Omit to discard after scoring.")
    args = ap.parse_args()

    if not (10.0 <= args.window < 15.0):
        print(f"WARNING: --window {args.window}s falls outside the model's "
              f"trained 10-15s duration bucket; predictions will be flagged "
              f"out-of-distribution.")

    save_dir = Path(args.save_clips) if args.save_clips else None

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"Listening on 0.0.0.0:{args.port} -- waiting for the XIAO ESP32-S3 ...")

    try:
        while True:
            conn, addr = srv.accept()
            print(f"Connection from {addr}")
            try:
                handle_connection(conn, args.window, save_dir)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
