/*
 * XIAO ESP32-S3 + INMP441 I2S microphone test / deployment firmware
 * -------------------------------------------------------------------
 * Wiring, I2S mic (see README.md for the full table):
 *   INMP441 VCC -> XIAO 3V3
 *   INMP441 GND -> XIAO GND
 *   INMP441 L/R -> XIAO GND   (selects LEFT channel output)
 *   INMP441 WS  -> XIAO D9  (GPIO8)
 *   INMP441 SCK -> XIAO D8  (GPIO7)
 *   INMP441 SD  -> XIAO D10 (GPIO9)
 *
 * Wiring, legacy analog condenser sensor module (only needed for
 * MODE_DUAL_COMPARE -- see README.md Section 5):
 *   Module VCC        -> XIAO 3V3 (NOT 5V/VBUS -- see README warning)
 *   Module GND        -> XIAO GND
 *   Module analog out -> XIAO D1 (GPIO2, ADC1-capable)
 *   Module DO (comparator, optional, unused by firmware) -> XIAO D3 (GPIO4)
 *
 * SAMPLE_RATE is fixed at 8000 Hz in every mode below because that is
 * what the trained ensemble model (src/inference/predict.py, src/dl/dl_dataset.py:
 * CLIP_SAMPLES = 80_000 = 10.0s @ 8kHz) actually expects. Capturing at
 * any other rate changes the numeric value of every spectral feature
 * the model was trained on, even after nominal resampling.
 *
 * Four run modes, selected below with CAPTURE_MODE:
 *   MODE_LEVEL_METER  (0) - prints RMS/peak dBFS twice a second. Use this
 *                           first to confirm wiring and clock config.
 *   MODE_RAW_STREAM   (1) - streams raw 16-bit PCM as binary over Serial.
 *                           Pair with tools/serial_to_wav.py to capture a
 *                           single test .wav file.
 *   MODE_WIFI_STREAM  (2) - deployment mode. Connects to Wi-Fi and streams
 *                           raw 16-bit PCM continuously over a TCP socket
 *                           to tools/wifi_audio_server.py, which buffers
 *                           10-15s windows and runs them through the full
 *                           trained ensemble (src/inference/predict.py) on the host.
 *                           Requires include/wifi_config.h -- copy
 *                           wifi_config.h.example and fill in your
 *                           network + host details before building.
 *   MODE_DUAL_COMPARE (3) - sensor-migration validation mode. Reads the
 *                           INMP441 (I2S) and the legacy analog condenser
 *                           module (ADC) in the same loop iteration and
 *                           streams both, interleaved as a stereo PCM
 *                           frame (L=INMP441, R=analog module), to
 *                           tools/dual_mic_compare.py, which saves a
 *                           stereo WAV and prints/plots a feature-by-
 *                           feature comparison. Also requires
 *                           wifi_config.h.
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>

#define MODE_LEVEL_METER  0
#define MODE_RAW_STREAM   1
#define MODE_WIFI_STREAM  2
#define MODE_DUAL_COMPARE 3
#define CAPTURE_MODE MODE_LEVEL_METER

#if CAPTURE_MODE == MODE_WIFI_STREAM || CAPTURE_MODE == MODE_DUAL_COMPARE
#include <WiFi.h>
#include "wifi_config.h"   // WIFI_SSID, WIFI_PASSWORD, SERVER_HOST, SERVER_PORT
#endif

// ---- I2S pin map (INMP441) ----
#define I2S_SCK_PIN 7   // D8  -> INMP441 SCK (bit clock)
#define I2S_WS_PIN  8   // D9  -> INMP441 WS  (word select / L-R clock)
#define I2S_SD_PIN  9   // D10 -> INMP441 SD  (serial data, mic -> MCU)

// ---- Analog pin map (legacy condenser sensor module, MODE_DUAL_COMPARE only) ----
#define ANALOG_MIC_PIN 2   // D1 (GPIO2, ADC1_CH1) -> module's analog output
#define COMPARATOR_PIN 4   // D3 (GPIO4)           -> module's digital threshold pin (read-only, not streamed)

#define I2S_PORT      I2S_NUM_0
#define SAMPLE_RATE   8000                        // must match the training pipeline exactly
#define DMA_BUF_COUNT 8
#define DMA_BUF_LEN   256                       // samples per DMA buffer
#define READ_SAMPLES  DMA_BUF_LEN                // int32_t words per i2s_read() call
#define FULL_SCALE_24BIT 8388608.0f              // 2^23, INMP441 full scale

static int32_t i2s_buf[READ_SAMPLES];

static void i2s_install() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,   // INMP441 sends 24-bit data left-justified in a 32-bit slot
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,    // L/R pin tied to GND -> mic drives the left slot
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = DMA_BUF_COUNT,
    .dma_buf_len = DMA_BUF_LEN,
    .use_apll = true,                               // audio PLL: lower jitter, more accurate sample rate
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pins = {
    .mck_io_num = I2S_PIN_NO_CHANGE,
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN
  };

  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
  i2s_zero_dma_buffer(I2S_PORT);
}

#if CAPTURE_MODE == MODE_WIFI_STREAM || CAPTURE_MODE == MODE_DUAL_COMPARE
static WiFiClient client;

static void ensure_wifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("Connecting to Wi-Fi \"%s\" ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.printf("\nWi-Fi connected, IP: %s\n", WiFi.localIP().toString().c_str());
}

static void ensure_server() {
  if (client.connected()) return;
  Serial.printf("Connecting to host server %s:%d ...\n", SERVER_HOST, SERVER_PORT);
  while (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("  server connect failed, retrying in 2s");
    delay(2000);
  }
  Serial.println("Connected to host server, streaming...");
}

static void send_all(const uint8_t *data, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    if (!client.connected()) return;   // drop this chunk; ensure_server() will redial next loop()
    size_t n = client.write(data + sent, len - sent);
    if (n == 0) { delay(1); continue; }
    sent += n;
  }
}
#endif

void setup() {
  Serial.begin(115200);
#if CAPTURE_MODE == MODE_LEVEL_METER
  delay(1500);   // give the USB-CDC serial monitor time to attach
  Serial.println();
  Serial.println("XIAO ESP32-S3 + INMP441 -- I2S level meter test");
  Serial.printf("Sample rate: %d Hz | Pins SCK=%d WS=%d SD=%d\n",
                SAMPLE_RATE, I2S_SCK_PIN, I2S_WS_PIN, I2S_SD_PIN);
#elif CAPTURE_MODE == MODE_WIFI_STREAM
  delay(1500);
  Serial.println("XIAO ESP32-S3 + INMP441 -- Wi-Fi deployment stream");
  ensure_wifi();
  ensure_server();
#elif CAPTURE_MODE == MODE_DUAL_COMPARE
  delay(1500);
  Serial.println("XIAO ESP32-S3 -- dual-mic comparison stream (INMP441 vs analog module)");
  analogReadResolution(12);                       // 0-4095 over 0-3.3V
  analogSetPinAttenuation(ANALOG_MIC_PIN, ADC_11db);
  pinMode(COMPARATOR_PIN, INPUT);
  ensure_wifi();
  ensure_server();
#endif
  i2s_install();
}

void loop() {
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, i2s_buf, sizeof(i2s_buf), &bytes_read, portMAX_DELAY);
  int samples = bytes_read / sizeof(int32_t);
  if (samples == 0) return;

#if CAPTURE_MODE == MODE_LEVEL_METER
  int64_t sum_sq = 0;
  int32_t peak = 0;
  for (int i = 0; i < samples; i++) {
    int32_t s24 = i2s_buf[i] >> 8;              // arithmetic shift: sign-extends the 24-bit sample
    sum_sq += (int64_t)s24 * (int64_t)s24;
    int32_t a = s24 < 0 ? -s24 : s24;
    if (a > peak) peak = a;
  }
  float rms = sqrtf((float)((double)sum_sq / samples));
  float rms_dbfs  = 20.0f * log10f(rms  / FULL_SCALE_24BIT + 1e-9f);
  float peak_dbfs = 20.0f * log10f((float)peak / FULL_SCALE_24BIT + 1e-9f);
  Serial.printf("RMS: %6.1f dBFS   Peak: %6.1f dBFS   (n=%d)\n", rms_dbfs, peak_dbfs, samples);

#elif CAPTURE_MODE == MODE_RAW_STREAM
  static int16_t pcm16[READ_SAMPLES];
  for (int i = 0; i < samples; i++) {
    int32_t s24 = i2s_buf[i] >> 8;
    pcm16[i] = (int16_t)(s24 >> 8);             // drop 24-bit -> 16-bit for a compact, standard PCM stream
  }
  Serial.write((uint8_t *)pcm16, samples * sizeof(int16_t));

#elif CAPTURE_MODE == MODE_WIFI_STREAM
  static int16_t pcm16[READ_SAMPLES];
  for (int i = 0; i < samples; i++) {
    int32_t s24 = i2s_buf[i] >> 8;
    pcm16[i] = (int16_t)(s24 >> 8);
  }
  ensure_wifi();
  ensure_server();
  send_all((const uint8_t *)pcm16, samples * sizeof(int16_t));

#elif CAPTURE_MODE == MODE_DUAL_COMPARE
  // Interleaved stereo frame: L = INMP441 (digital, true per-sample), R = analog
  // module (sampled back-to-back with analogRead() right after the I2S DMA block
  // returns). Both channels cover the same ~32ms window at 8000 samples/sec;
  // they are not phase-locked sample-for-sample, but that's enough resolution
  // for comparing RMS/spectral statistics between the two sensors, not for
  // sample-accurate DSP fusion.
  static int16_t stereo[READ_SAMPLES * 2];
  for (int i = 0; i < samples; i++) {
    int32_t s24 = i2s_buf[i] >> 8;
    int16_t i2s16 = (int16_t)(s24 >> 8);

    int adc_raw = analogRead(ANALOG_MIC_PIN);              // 0-4095 over 0-3.3V, ~10-20us
    int16_t analog16 = (int16_t)((adc_raw - 2048) * 16);   // recenter and rescale to signed 16-bit

    stereo[i * 2]     = i2s16;
    stereo[i * 2 + 1] = analog16;
  }
  ensure_wifi();
  ensure_server();
  send_all((const uint8_t *)stereo, samples * 2 * sizeof(int16_t));
#endif
}
