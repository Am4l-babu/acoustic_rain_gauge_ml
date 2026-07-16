"""
Capture the raw 16-bit PCM stream produced by main.cpp (CAPTURE_MODE =
MODE_RAW_STREAM) over serial and save it as a standard mono WAV file.

Usage:
    python serial_to_wav.py --port COM5 --seconds 10 --out capture.wav

Requires: pyserial   (pip install pyserial)

Notes:
  - Flip CAPTURE_MODE to MODE_RAW_STREAM in src/main.cpp and re-upload
    before running this script; MODE_LEVEL_METER mode sends text, not PCM.
  - Baud rate must match Serial.begin() in main.cpp (115200 by default).
    That is plenty of headroom for 16 kHz mono 16-bit audio
    (16000 * 2 bytes = 32000 B/s = 256 kbps, well under 115200's ~11.5 kB/s...
    actually see README for the recommended baud rate at 44.1/48 kHz).
"""

import argparse
import struct
import wave

import serial


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True, help="Serial port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate", type=int, default=16000, help="Must match SAMPLE_RATE in main.cpp")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="capture.wav")
    args = ap.parse_args()

    n_bytes = int(args.rate * args.seconds) * 2  # 16-bit = 2 bytes/sample

    print(f"Opening {args.port} @ {args.baud} baud ...")
    with serial.Serial(args.port, args.baud, timeout=5) as ser:
        ser.reset_input_buffer()
        print(f"Recording {args.seconds:.1f}s ({n_bytes} bytes) -> {args.out}")
        data = bytearray()
        while len(data) < n_bytes:
            chunk = ser.read(n_bytes - len(data))
            if not chunk:
                print("WARNING: serial read timed out, stopping early.")
                break
            data.extend(chunk)

    # Drop a trailing odd byte if the stream was cut mid-sample.
    if len(data) % 2:
        data = data[:-1]

    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(args.rate)
        wf.writeframes(bytes(data))

    n_samples = len(data) // 2
    peak = max(abs(s) for s in struct.unpack(f"<{n_samples}h", bytes(data))) if n_samples else 0
    print(f"Wrote {args.out}: {n_samples} samples, {n_samples / args.rate:.2f}s, peak={peak}/32767")


if __name__ == "__main__":
    main()
