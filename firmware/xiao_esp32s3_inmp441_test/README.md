# XIAO ESP32-S3 + INMP441 — I2S Microphone Bring-Up, Deployment & Sensor-Migration Test

A PlatformIO project covering the full lifecycle of moving this project's
acoustic sensing to a Seeed Studio XIAO ESP32-S3 + INMP441 MEMS digital
microphone: wiring/bring-up, capturing test `.wav` files, streaming live
audio to the trained ensemble model over Wi-Fi, and validating the switch
away from the project's original analog condenser sensor module before
retiring it.

For the full engineering handbook (specs, architecture, PCB guidelines,
troubleshooting, 100+ project ideas), see the companion documentation.

## 1. Wiring

### 1a. INMP441 (the new digital mic)

| INMP441 pin | XIAO ESP32-S3 pin | GPIO | Notes |
|---|---|---|---|
| VCC | 3V3 | — | 1.8–3.3 V; use the 3V3 rail, not VBUS (5 V) |
| GND | GND | — | Also ties L/R reference — see below |
| L/R | GND | — | Grounded = LEFT channel output (matches `I2S_CHANNEL_FMT_ONLY_LEFT` in code) |
| WS | D9 | GPIO8 | Word Select / LRCLK |
| SCK | D8 | GPIO7 | Bit clock (BCLK) |
| SD | D10 | GPIO9 | Serial data, mic → MCU (one-directional) |

```
        XIAO ESP32-S3                       INMP441
       ┌───────────────┐                 ┌───────────────┐
       │            3V3●─────────────────●VCC            │
       │            GND●──────────┬──────●GND             │
       │                          └──────●L/R  (LEFT sel) │
       │      D8 (GPIO7,SCK)●─────────────●SCK             │
       │      D9 (GPIO8,WS) ●─────────────●WS              │
       │      D10(GPIO9,SD) ●─────────────●SD              │
       └───────────────┘                 └───────────────┘
```

These three GPIOs (7/8/9) were chosen because they avoid every
boot-strapping pin (GPIO0, GPIO3, GPIO45, GPIO46) and the native-USB pins
(GPIO19/20, wired internally to the USB-C port, not exposed on the header),
so the serial monitor and reflashing always stay usable. D0–D7 remain free
for a tipping-bucket reed switch, status LED, SD card, etc.

### 1b. Legacy analog condenser sensor module (only for `MODE_DUAL_COMPARE`, Section 5)

This is the cheap electret + LM393-comparator module (e.g. the Robu.in
"Small Microphone Sound Sensor Module", 5V-rated, analog out + comparator
threshold out) the project has been using so far. Wire it in **alongside**
the INMP441 only when you specifically want to run the side-by-side
sensor-migration comparison in Section 5 — it's not needed for normal
bring-up or deployment.

| Module pin | XIAO ESP32-S3 pin | GPIO | Notes |
|---|---|---|---|
| VCC | **3V3**, not the module's rated 5V | — | See warning below |
| GND | GND | — | Common ground with everything else |
| Analog out (AO) | D1 | GPIO2 (ADC1_CH1) | Read via `analogRead()`, 12-bit, 0–3.3V range |
| Comparator out (DO) | D3 | GPIO4 | Optional; a binary threshold flag, not used by the firmware's streaming payload |

```
        XIAO ESP32-S3                    Analog sensor module
       ┌───────────────┐                 ┌───────────────────┐
       │            3V3●─────────────────●VCC (NOT 5V/VBUS!)  │
       │            GND●─────────────────●GND                 │
       │      D1 (GPIO2)●─────────────────●AO (analog out)     │
       │      D3 (GPIO4)●─────────────────●DO (comparator, opt)│
       └───────────────┘                 └───────────────────┘
```

**Warning — power it from 3V3, not 5V:** the module's own analog output
rides on whatever rail powers it. Powering it from XIAO's 5V/VBUS pin can
put a signal on GPIO2 that exceeds the ESP32-S3 ADC's 3.3V absolute
maximum, risking pin damage. Most LM393-based modules like this one work
fine at 3.3V; if yours specifically requires 5V, add a resistor divider
(e.g. two 10kΩ resistors) between AO and GPIO2 instead of powering it at
3.3V, so the ADC never sees more than 3.3V.

## 2. Build & upload with PlatformIO

```bash
# from this directory (firmware/xiao_esp32s3_inmp441_test)
pio run                 # build
pio run -t upload       # flash over USB-C
pio device monitor      # open serial monitor (115200 baud)
```

Or in VS Code with the PlatformIO extension: open this folder, click the
checkmark (Build), then the arrow (Upload) in the status bar.

Put the XIAO into bootloader mode manually only if upload fails to
auto-reset: hold **BOOT**, tap **RESET**, release **BOOT**.

## 3. Four run modes

Edit `CAPTURE_MODE` near the top of `src/main.cpp`, then re-upload.

| Mode | Value | What it does | Use for |
|---|---|---|---|
| `MODE_LEVEL_METER` | `0` (default) | Prints RMS/peak dBFS ~30x/sec over serial | First bring-up: confirm wiring, clocking, that the mic responds to sound |
| `MODE_RAW_STREAM` | `1` | Streams raw 16-bit PCM as binary over serial | Capturing one `.wav` file by hand, via `tools/serial_to_wav.py` |
| `MODE_WIFI_STREAM` | `2` | Connects to Wi-Fi, streams raw 16-bit PCM continuously over TCP | Deployment: feeding the live trained ensemble via `tools/wifi_audio_server.py` (Section 4) |
| `MODE_DUAL_COMPARE` | `3` | Connects to Wi-Fi, streams interleaved stereo PCM (INMP441 + analog module) over TCP | Sensor-migration validation: comparing the old and new mic via `tools/dual_mic_compare.py` (Section 5) |

All four modes capture at a fixed **8000 Hz**, matching what the trained
model actually expects (`src/dl/dl_dataset.py`'s `CLIP_SAMPLES = 80_000` =
10.0s @ 8kHz). This changed from an earlier 16kHz default once the model's
real input rate was confirmed — keep it at 8000 unless you retrain.

### Expected level-meter output

Quiet room: RMS around -55 to -45 dBFS, peak spiking on any noise.
Tap the mic capsule gently and you should see peak jump toward -20 to 0 dBFS.

```
RMS:  -48.2 dBFS   Peak:  -31.5 dBFS   (n=256)
RMS:  -47.9 dBFS   Peak:  -30.8 dBFS   (n=256)
RMS:  -12.3 dBFS   Peak:   -2.1 dBFS   (n=256)   <- clap
```

If RMS sits near 0 dBFS constantly or the numbers never move, see
Troubleshooting below.

### Capturing a WAV file

```bash
# 1. Set CAPTURE_MODE to MODE_RAW_STREAM in main.cpp, then:
pio run -t upload

# 2. Find your serial port first (Windows: check Device Manager, or):
pio device list

# 3. Capture 10 seconds of audio:
pip install pyserial
python tools/serial_to_wav.py --port COM5 --seconds 10 --out capture.wav
```

Play `capture.wav` back in any audio player to confirm real sound was
recorded (not silence or noise), then bring it into the main ML pipeline.

## 4. Deployment — streaming live to the trained ensemble over Wi-Fi

This is the actual "run the model on real audio" path. The XIAO does **not**
run the model itself — the trained ensemble (CNN + LSTM + Transformer +
XGBoost + stacker, R²=0.5429) is a full PyTorch/XGBoost pipeline that has no
practical microcontroller runtime. Instead, the XIAO is a Wi-Fi audio sensor
node: it streams raw PCM continuously to a small Python server running on
your PC, which buffers it into windows and calls the existing
`src/inference/predict.py: predict_ensemble()` unmodified — full accuracy, zero model
conversion work.

```
 INMP441 --I2S--> XIAO ESP32-S3 --Wi-Fi/TCP--> wifi_audio_server.py --> predict_ensemble()
                  (MODE_WIFI_STREAM)            (buffers 10-15s windows)   (src/inference/predict.py, unchanged)
```

**Why a 10-15 second window:** `src/inference/predict.py` classifies clip duration
into buckets and only the `"10-15s"` bucket matches what the model was
trained on (`TRAINED_DURATION_CATEGORY`); anything else is flagged
`out_of_distribution` in the result. `wifi_audio_server.py` defaults its
window to 12.0s for this reason — don't shrink it below 10s or grow it past
15s without also checking `src/dl/dl_dataset.py`/`src/features/data_cleaning.py`'s duration
logic.

### Setup

```bash
# 1. Create your Wi-Fi config (never committed — see .gitignore)
cp include/wifi_config.h.example include/wifi_config.h
# edit include/wifi_config.h: WIFI_SSID, WIFI_PASSWORD, SERVER_HOST (your PC's LAN IP), SERVER_PORT

# 2. Flip the mode and upload
#    In src/main.cpp: #define CAPTURE_MODE MODE_WIFI_STREAM
pio run -t upload

# 3. Start the receiver on your PC (run from the repo root so src/ imports resolve,
#    or let the script's own sys.path handling find it — either works)
python firmware/xiao_esp32s3_inmp441_test/tools/wifi_audio_server.py --port 9494

# Optional: keep every scored clip for building a field dataset later
python firmware/xiao_esp32s3_inmp441_test/tools/wifi_audio_server.py --port 9494 --save-clips data/field_captures/
```

Power-cycle or reset the XIAO after the server is listening. It connects to
Wi-Fi, dials the server, and streams continuously; the server prints one
line per completed window:

```
Listening on 0.0.0.0:9494 -- waiting for the XIAO ESP32-S3 ...
Connection from ('192.168.1.42', 51823)
Client connected. Buffering 12.0s windows (192000 bytes) at 8000 Hz.
[20260716_091023] 12.0s  P(rain)=0.812  est=1.204mm  -> RAINY
[20260716_091035] 12.0s  P(rain)=0.043  est=0.000mm  -> dry
```

Find your PC's LAN IP for `SERVER_HOST` with `ipconfig` (Windows, look for
IPv4 Address) — both devices must be on the same Wi-Fi network.

## 5. Sensor migration — comparing the old analog mic against the INMP441

**Why bother:** the trained ensemble (R²=0.5429) was built on audio captured
through the project's original analog condenser sensor module, not the
INMP441. That module is a cheap electret + LM393 comparator with
board-to-board-variable gain, an uncontrolled frequency response, and a
higher noise floor — very different electrical behavior from the INMP441's
flat, calibrated, fixed-sensitivity digital output. Swapping sensors can
shift the numeric value of every feature the model was trained on (RMS,
spectral centroid, MFCCs, ...) even for identical real-world sound. Before
retiring the old module, capture both at once and check how much its
retirement actually changes what the model sees.

```
INMP441 (I2S) ──┐
                 ├──> XIAO ESP32-S3 ──Wi-Fi/TCP──> dual_mic_compare.py ──> stereo .wav
Analog module ──┘     (MODE_DUAL_COMPARE)           (feature comparison + plot)
   (ADC)
```

Wire the analog module alongside the INMP441 per Section 1b (D1=analog out,
3V3 power — **not** 5V). The firmware reads the INMP441 via I2S DMA and the
analog module via `analogRead()` in the same loop iteration, then streams
them as one interleaved stereo frame (left=INMP441, right=analog module).

### Running the comparison

```bash
# 1. Wire the analog module per Section 1b, in addition to the INMP441.

# 2. Flip the mode and upload
#    In src/main.cpp: #define CAPTURE_MODE MODE_DUAL_COMPARE
pio run -t upload

# 3. Start the comparison receiver (reuses wifi_config.h's SERVER_PORT)
python firmware/xiao_esp32s3_inmp441_test/tools/dual_mic_compare.py --port 9494 --seconds 10
```

Make some representative sound during the 10-second capture (tap near both
mics, run water, or — best of all — record during light real rain). The
script then:

1. Saves a stereo WAV (`dual_capture.wav`): left channel = INMP441, right = analog module — play it back and toggle channels to listen to each sensor individually.
2. Runs the exact same feature extraction the training pipeline uses (`data_cleaning._extract_features`, no reimplementation) on each channel separately.
3. Prints a side-by-side table:

   ```text
   ========================================================================
   Feature                    INMP441 (I2S)     Analog module         Ratio
   ========================================================================
   rms                              0.04231           0.01187         3.564
   peak                             0.31200           0.09850         3.168
   spectral_centroid             1834.20200        912.44500         2.011
   spectral_bandwidth            1120.55000        640.12000         1.751
   zero_crossing_rate               0.08120          0.03410         2.381
   mfcc_0                         -412.30000       -198.77000         2.074
   ...
   ========================================================================
   ```

4. Saves `dual_capture_comparison.png` with an overlaid waveform plot and an overlaid FFT magnitude spectrum, so you can see visually where the two sensors' frequency response diverges.

**Reading the result:** ratios far from 1.0 — especially on `spectral_centroid`
and the `mfcc_*` features, which drive most of the model's decision —
mean the two sensors disagree enough that the existing model likely
**will not** generalize correctly to INMP441 audio as-is. In that case,
plan to either recollect a labeled training set with the INMP441 (recommended,
matches Section 4's `--save-clips` option) or derive/validate a correction
scaling before trusting `predict_ensemble()` output on live INMP441 streams.
A small, consistent ratio across most features is a reasonable, though not
definitive, signal that the old model is safe to keep using during a
transition period.

## 6. Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| No serial output at all | Wrong port / board not flashed / USB CDC not enabled | Check `pio device list`; confirm `ARDUINO_USB_CDC_ON_BOOT=1` in `platformio.ini` |
| RMS constant near 0 dBFS (clipped) | WS/SCK swapped, or SD floating/shorted | Re-check wiring against the table above |
| RMS constant at very low noise floor, no response to sound | L/R pin not grounded (mic outputs on the other channel) | Confirm L/R → GND, and code uses `I2S_CHANNEL_FMT_ONLY_LEFT` |
| Garbage / static-y audio | Sample rate mismatch between firmware and `serial_to_wav.py --rate` | Make sure `--rate` matches `SAMPLE_RATE` in `main.cpp` (8000 by default) |
| Upload fails / port busy | Serial monitor left open, or board stuck in a boot loop | Close monitor before `pio run -t upload`; manual BOOT+RESET if needed |
| Build fails on `driver/i2s.h` | Platform version drifted from the pin in `platformio.ini` | Keep `platform = espressif32@6.9.0` (legacy I2S driver); see main docs for the ESP-IDF 5.x `i2s_std.h` alternative |
| Build fails: `wifi_config.h: No such file` | Config template not copied yet | `cp include/wifi_config.h.example include/wifi_config.h` and fill it in (Section 4) |
| XIAO connects to Wi-Fi but never reaches the server | Wrong `SERVER_HOST` IP, PC firewall blocking the port, or devices on different networks/VLANs (e.g. guest Wi-Fi) | Verify `SERVER_HOST` via `ipconfig`; temporarily allow the port through the OS firewall; confirm both devices show the same subnet |
| Server prints `out_of_distribution: True` on every clip | `--window` outside 10-15s, or firmware `SAMPLE_RATE` isn't 8000 | Keep `--window` at its 12.0s default and `SAMPLE_RATE` at 8000 in both `main.cpp` and `wifi_audio_server.py` |
| Predictions look constant/wrong regardless of real rain | Mic capturing silence/noise (wiring), or `SAMPLE_RATE` mismatch corrupting feature values | Re-verify with `MODE_LEVEL_METER` first; confirm 8000 Hz end-to-end |
| ADC (analog module) reads pinned at 0 or 4095 | Analog out wired to the wrong pin, module not powered, or module powered from 5V with no divider (may indicate pin damage — stop and re-check before reusing GPIO2) | Re-check Section 1b wiring; confirm 3V3 power; test the module's AO pin with a multimeter first |
| Dual-compare feature ratios wildly inconsistent between repeated runs | Test stimulus not actually simultaneous/comparable between runs (e.g. different distance to each mic) | Keep both sensors physically close together and equidistant from the sound source for a fair comparison |
| `dual_mic_compare.py` fails to import `data_cleaning` | Script run from an unexpected working directory | It auto-adds `REPO_ROOT/src` to `sys.path`; if this still fails, confirm the repo layout hasn't moved relative to `firmware/xiao_esp32s3_inmp441_test/tools/` |

## 7. Files in this project

```
xiao_esp32s3_inmp441_test/
├── platformio.ini              Board/platform config, pinned for I2S API stability
├── include/
│   └── wifi_config.h.example    Template for Wi-Fi + server credentials (copy -> wifi_config.h)
├── src/
│   └── main.cpp                 Level-meter + raw-serial-stream + Wi-Fi deployment-stream +
│                                 dual-mic-compare firmware (CAPTURE_MODE selects which)
├── tools/
│   ├── serial_to_wav.py          Captures MODE_RAW_STREAM output into a single .wav
│   ├── wifi_audio_server.py      Receives MODE_WIFI_STREAM audio, scores it via src/inference/predict.py
│   └── dual_mic_compare.py       Receives MODE_DUAL_COMPARE audio, compares INMP441 vs the
│                                 analog module using src/features/data_cleaning.py's feature extraction
└── README.md                    This file
```
