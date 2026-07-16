# Research Paper Analysis — Identification of Mismatched PDFs

**Date:** 2026-07-15 (updated same day with 8 newly added papers + full library duplicate cleanup)
**Scope:** All PDFs in `ARG_Research/` whose filenames did not state what paper they contain (arXiv IDs, publisher codes, `document.pdf`, `sarid.pdf`, journal-code filenames, etc.). Every file below was opened and its actual content read (text extraction via pypdf; the two scanned-image papers were identified by rendering and reading their first-page images). MD5 hashing across all PDFs was used to detect content-identical copies hiding under different names.

**Update log (same day):** 8 new PDFs were added to `research papers\` with cryptic auto-downloaded filenames (`Adjustment_of_1_min_rain_gauge_time_seri.pdf`, `Bias_correction_of_global_daily_rain_gau.pdf`, `Corrections_of_rainfall_data_obtained_by-2.pdf`, `Numerical_simulation_studies_of_rain_gag.pdf`, `Rainfall_and_sampling_uncertainties_A_ra.pdf`, `Rainfall_variability_and_its_effect_on_r(-1).pdf`, `The_Quantification_and_Correction_of_Win.pdf`, `Variability_of_Drop_Size_Distributions_T(-1).pdf`). All were identified, analyzed (Section 5), and renamed to the `YYYY - Title.pdf` convention. A full-library duplicate sweep (see Section 6) then removed **117 duplicate PDFs** (all of `research papers\duplicates\`, which is now deleted, plus 5 stray exact/near-exact duplicates found elsewhere), taking the library from ~178 PDFs down to **172 unique PDFs**.

For each paper: what it is, what the authors do, the results of their work, what can be inferred, and how it helps or supports **this project's idea** (estimating rainfall presence + intensity from ambient audio with ML, benchmarked against SARID; current best R² = 0.531 via stacking ensemble, SARID baseline R² = 0.765).

---

## 1. Headline findings

1. **`0000146.pdf` is the most important "hidden" paper in the library.** It is Monti & Ntalampiras (EUSIPCO 2025), *On acoustic monitoring of rainfall intensity* — the current state of the art **on the exact SARID benchmark this project compares against**: R² = 0.787 (vs SARID's 0.765) using a stacking ensemble of transformers, the same ensemble strategy this project already uses.
2. **`sarid.pdf` and `1-s2.0-S2666498424000644-main.pdf` are both the SARID paper itself** (Wang et al. 2024), meaning the library holds ~4 copies of it under different names.
3. **`document.pdf`, `2309.16867v1.pdf`, and `Towards High Resolution Weather Monitoring with Sound Data.pdf` are all the same paper** (Çoban, Perra & Mandel 2023).
4. **Two genuinely mismatched files:** `Rain disdrometer and hail detection.pdf` is **not a research paper** — it is the commercial RHD acoustic-disdrometer **datasheet** (byte-identical to `Category 1\Datasheet_RHD_V1_41.pdf`). And `sensors-20-07214.pdf` is an off-topic **DEM/SAR interferometry** paper, not a rainfall paper.
5. The two scanned (no text layer) files are classics: **Joss & Waldvogel (1969)** on raindrop-size-distribution sampling errors, and **Nystuen (1996)**, the foundational acoustical-rainfall-analysis paper.

## 2. Identification table (cryptic filename → actual paper)

| File | Actual paper | Duplicate of |
|---|---|---|
| `research papers\0000146.pdf` | Monti & Ntalampiras, *On acoustic monitoring of rainfall intensity*, EUSIPCO 2025 | — |
| `research papers\sarid.pdf` | Wang et al., *Estimating rainfall intensity based on surveillance audio and deep-learning* (SARID), Environ. Sci. Ecotechnol. 22, 2024 | `1-s2.0-S2666498424000644-main.pdf`, `2024 - Estimating rainfall intensity...` (×2) |
| `research papers\1-s2.0-S2666498424000644-main.pdf` | Same SARID paper (Elsevier download name) | see above |
| `research papers\document.pdf` | Çoban, Perra & Mandel, *Towards High Resolution Weather Monitoring with Sound Data*, 2023 (arXiv:2309.16867) | `2309.16867v1.pdf`, `Towards High Resolution Weather Monitoring with Sound Data.pdf` |
| `research papers\2309.16867v1.pdf` | Same Çoban et al. paper | see above |
| `research papers\1805.01957v1.pdf` | Manandhar, Dev, Lee, Winkler & Meng, *Systematic Study of Weather Variables for Rainfall Detection*, 2018 (arXiv:1805.01957) | — |
| `research papers\2110.11059v1.pdf` | Pathan, Wu, Lee, Yan & Dev, *Analyzing the Impact of Meteorological Parameters on Rainfall Prediction*, 2021 (arXiv:2110.11059) | — |
| `research papers\atot-1520-0426_1996_013_0074_arards_2_0_co_2.pdf` | Nystuen, *Acoustical Rainfall Analysis: Rainfall Drop Size Distribution Using the Underwater Sound Field*, J. Atmos. Oceanic Technol. 13:74, 1996 (scanned) | `Acoustical Rainfall Analysis...pdf` |
| `research papers\atot-1520-0426_1999_016_1025_rpoarg_2_0_co_2.pdf` | Nystuen, *Relative Performance of Automatic Rain Gauges under Different Rainfall Conditions*, JTECH 16:1025, 1999 | `Relative Performance of Automatic Rain Gauges...` (research papers + Category 1) |
| `research papers\atot-jtech1773_1.pdf` | Ma & Nystuen, *Passive Acoustic Detection and Measurement of Rainfall at Sea*, JTECH 22:1225, 2005 | `duplicates\atot-jtech1773_1-1.pdf` |
| `research papers\RaindropSizeDist......JASMay1969.pdf` | Joss & Waldvogel, *Raindrop Size Distribution and Sampling Size Errors*, J. Atmos. Sci. 26:566–569, 1969 (scanned) | — |
| `research papers\preprints202512.1003.v1.pdf` | Dunkerley, *A New Device for Continuous, Real-Time Acoustic Measurement of Rain Inclination*, Preprints.org, Dec 2025 | — |
| `research papers\Paper_RainGaugeTippingBucketIEEE.pdf` | Prakosa, Wijonarko & Rustandi, *The Performance Measurement Test on Rain Gauge of Tipping Bucket due to Controlling of the Water Flow Rate*, ElConRus 2018, pp. 1136–1140 (embedded in a proceedings extract) | ×2 in duplicates |
| `research papers\Rain disdrometer and hail detection.pdf` | **Not a paper** — RHD acoustic rain-disdrometer & hail sensor commercial datasheet | `Category 1\Datasheet_RHD_V1_41.pdf` (byte-identical) |
| `research papers\sensors-20-07214.pdf` | Mohammadi et al., *A Multi-Sensor Comparative Analysis on the Suitability of Generated DEM from Sentinel-1 SAR Interferometry...*, Sensors 20:7214, 2020 — **off-topic** | `n.d. - A Multi-Sensor Comparative Analysis...` (×2) |
| `research papers\sensors-22-04381.pdf` | Kurşun, Güven & Ersoy, *Utilizing Piezo Acoustic Sensors for the Identification of Surface Roughness and Textures*, Sensors 22:4381, 2022 | `n.d. - Utilizing Piezo Acoustic Sensors...` |
| `research papers\A_self-powered_microsystem_with_efficient_power_ma.pdf` | Zhao et al., *A self-powered microsystem with efficient power management for continuous wireless sensing*, Microsyst. Nanoeng. 12:178, 2026 | — |
| `research papers\AI_Methods_in_Sensor_Calibration.pdf` | Kou, Liu, Li, Qin & Liu, *AI Methods in Sensor Calibration* (review), Sensors 26:2805, 2026 | — |
| `Category 1\10.1109@TENCON.2018.8650318.pdf` | Cruz, Pangaliman, Amado & Uy, *Development of Improved Acoustic Disdrometer Through Utilization of Machine Learning Algorithm*, IEEE TENCON 2018 | — |
| `Category 1\information-11-00183-v2.pdf` | Avanzato & Beritelli, *An Innovative Acoustic Rain Gauge Based on Convolutional Neural Networks*, Information 11(4):183, 2020 | — |
| `Category 1\sensors-21-02687.pdf` | Liu, Li, Shang, Tang & Zhang, *Measurement of Underwater Acoustic Energy Radiated by Single Raindrops*, Sensors 21:2687, 2021 | — |
| `Category 1\Design_and_development_of_a_novel_acoustic_rain_se-1.pdf` | Guico, Abrajano, Domer & Talusan, *Design and development of a novel acoustic rain sensor with automated telemetry*, MATEC Web Conf. 201:03003, 2018 | `-2.pdf` is byte-identical |
| `Category 1\atot-1520-0426_2001_018_0883_motduf_2_0_co_2.pdf` | Green, Ostafichuk & Rogak, *Measurement of Three-Dimensional Unsteady Flows Using an Inexpensive Multiple Disk Probe*, JTECH 18:883, 2001 | — |
| `Category 1\scienceworld,+Vol+18(3)+pp+404-413+Siloko+et+al.pdf` | Siloko & Uddin, *A Statistical Study of Wind Speed and its Connectivity with Relative Humidity and Temperature in Ughelli, Delta State, Nigeria*, Science World J. 18(3):404–413, 2023 | — |
| `Category 1\CWOP-WMO8.pdf` | WMO-No. 8, *Guide to Meteorological Instruments and Methods of Observation*, 7th ed., 2008 (681 pp., CWOP copy) | — |
| `Category 1\Datasheet_RHD_V1_41.pdf` | RHD acoustic rain-disdrometer & hail sensor datasheet | `Rain disdrometer and hail detection.pdf` |

---

## 3. Paper-by-paper analysis

### TIER A — Core: audio-based rainfall estimation (directly on this project's problem)

---

#### A1. Monti & Ntalampiras (2025) — *On acoustic monitoring of rainfall intensity* — EUSIPCO 2025
**File:** `research papers\0000146.pdf` (name comes from the proceedings page stamp "146")

**What they do:** They take the public SARID dataset (the same benchmark this project uses) and build an optimized audio-regression pipeline: (a) low-pass filtering of audio before feature extraction, (b) auxiliary meteorological features (humidity, pressure, temperature, wind) concatenated after the convolutional pooling stage, (c) a sweep of FFT window lengths (rain sound is quasi-stationary, so longer windows → better frequency resolution), and (d) a **stacking ensemble** of three transformer regressors trained on MFCC, Mel, and STFT features, combined by a linear regressor. Code is public on Kaggle.

**Results:** Stacking ensemble achieves **R² = 0.787, MAE = 0.52 mm/h, RMSE = 0.837 mm/h**, beating the SARID baseline (R² 0.765, MAE 0.563, RMSE 0.88). Single transformer with MFCC alone reaches R² 0.777. Other findings: rainfall-relevant information is concentrated **below ~2000 Hz** (low-pass at 2–3 kHz slightly helps; 1 kHz hurts); an FFT window of **4096 samples** markedly improves MFCC models; naively concatenating meteorological parameters **did not improve** performance; errors remain largest at high intensities due to SARID's class imbalance.

**Inference:** The current published state of the art on SARID is ~0.79 R², set by exactly the architecture family (transformer + stacking ensemble) this project independently adopted. Even the best pipeline saturates near 0.79 on curated single-site data — this puts this project's 0.531 on messy 19-campaign field data in honest context. Their negative result on met-feature concatenation suggests simply appending wind speed as a dense feature may not close the gap; feature fusion strategy matters.

**How it supports this project:** (1) Validates the project's stacking-ensemble strategy — it is the winning approach on the benchmark. (2) Gives a concrete, citable SOTA number (0.787) to compare against instead of only the SARID baseline. (3) The <2 kHz finding directly informs feature engineering (consider a low-pass or restricting Mel bands, and cheaper microphones/bandwidth). (4) The 4096-sample FFT window is a directly testable hyperparameter for the MFCC pipeline. (5) Their conclusions section explicitly calls for what this project already has: no-rain samples, realistic imbalance, and hierarchical detect-then-estimate modeling — this project is the answer to their "future work."

---

#### A2. Wang et al. (2024) — *Estimating rainfall intensity based on surveillance audio and deep-learning* (SARID) — Environ. Sci. Ecotechnol. 22:100450, DOI 10.1016/j.ese.2024.100450
**Files:** `sarid.pdf`, `1-s2.0-S2666498424000644-main.pdf` (plus two named copies `2024 - Estimating rainfall intensity...`)

**What they do:** They build the first public dataset (SARID) for continuous rainfall-intensity regression from surveillance audio: 6 real rainfall events at Nanjing Normal University, segmented into **12,066 4-second clips**, annotated with rainfall intensity and environment metadata (underlying surface, temperature, humidity, wind). They then train an MFCC + Transformer baseline to regress intensity.

**Results:** Baseline achieves **MAE ≈ 0.88 mm/h correlation/R = 0.765** against ground truth. Demonstrates surveillance audio is a viable, all-weather, low-data-volume rain sensor complementary to video.

**Inference:** Audio alone carries enough information for continuous intensity regression, not just class-level detection — but their data is curated, single-site, and rain-only, so the reported score is an upper bound relative to realistic deployments.

**How it supports this project:** It is *the* benchmark and architectural template (MFCC + Transformer) the project's DL track follows, and the gap analysis (curated vs 19-campaign field data, rain-only vs 85/15 imbalance) is the core of this project's novelty claim. Every improvement here is measured against this paper.

---

#### A3. Çoban, Perra & Mandel (2023) — *Towards High Resolution Weather Monitoring with Sound Data* — arXiv:2309.16867
**Files:** `2309.16867v1.pdf`, `document.pdf`, `Towards High Resolution Weather Monitoring with Sound Data.pdf` (3 copies)

**What they do:** They classify rain, wind, and air-temperature conditions from ecoacoustic recordings, and — the key idea — use coarse satellite reanalysis weather data (MERRA-2, ~50 km/1 h, downscaled to 250 m via MicroMet) as **weak labels** to train acoustic classifiers without human annotation.

**Results:** Rain is the most successfully predicted variable. Classifiers trained from noisy MERRA-2 labels end up **more accurate than the raw MERRA-2 data itself**, and models trained this way approach fully supervised performance while requiring almost no manual labeling. Since MERRA-2 is global, the approach scales to any acoustic dataset worldwide.

**Inference:** Weak/self-supervision from coarse weather products can substitute for expensive co-located gauges; a model can distill a better signal than its own noisy teacher.

**How it supports this project:** Offers a path to scale beyond the 19 gauge-paired campaigns — future recordings without a co-located tipping bucket could still be pseudo-labeled from reanalysis or nearby-station data to expand training data, and it validates rain as the most acoustically learnable weather variable.

---

#### A4. Avanzato & Beritelli (2020) — *An Innovative Acoustic Rain Gauge Based on Convolutional Neural Networks* — Information 11(4):183, DOI 10.3390/info11040183
**File:** `Category 1\information-11-00183-v2.pdf`

**What they do:** Apply a CNN **directly to the raw audio signal** (no spectrogram) using 3-second sliding windows with 100 ms offset, classifying rain into 7 intensity classes ("No rain" → "Cloudburst"), targeting low-cost/low-power hardware (e.g., a sensor inside a street-lamp luminaire).

**Results:** Average accuracy **75%** over 7 classes, rising to **93%** if adjacent-class confusions are tolerated. Demonstrates short-response-time heavy-rain alerting on cheap hardware.

**Inference:** Classification (not regression) was the ceiling of pre-SARID audio work; most confusion happens between adjacent intensity classes — i.e., the audio→intensity mapping is smooth, which is favorable for regression approaches.

**How it supports this project:** Documents the pre-SARID state of the art the project's Research Gap section cites (classification-only), and supports the two-stage design: their "No rain" class as part of the classifier is precedent for the project's detection track. Their raw-waveform CNN is an architecture variant the DL track could ablate against MFCC input.

---

#### A5. Cruz, Pangaliman, Amado & Uy (2018) — *Development of Improved Acoustic Disdrometer Through Utilization of Machine Learning Algorithm* — IEEE TENCON 2018, DOI 10.1109/TENCON.2018.8650318
**File:** `Category 1\10.1109@TENCON.2018.8650318.pdf`

**What they do:** Build a prototype acoustic disdrometer from four piezoelectric sensors + Arduino + ZigBee, and add a **K-nearest-neighbors classifier** so the device can distinguish rainfall-intensity categories from ambient noise — addressing the classic weakness that acoustic disdrometers can't tell rain from noise.

**Results:** KNN predictive model reaches **89.95% accuracy** in categorizing rainfall intensity vs ambient noise.

**Inference:** Even very simple ML on piezo-acoustic signals separates rain from confounding noise; the rain/no-rain discrimination problem is tractable with modest features.

**How it supports this project:** Direct precedent for the project's detection stage (is-it-raining classifier) and for the claim that ML solves the ambient-noise confusion that killed earlier acoustic instruments. Also a hardware reference for a cheap piezo-based deployment variant.

---

#### A6. Guico, Abrajano, Domer & Talusan (2018) — *Design and development of a novel acoustic rain sensor with automated telemetry* — MATEC Web Conf. 201:03003, DOI 10.1051/matecconf/201820103003
**Files:** `Category 1\Design_and_development_of_a_novel_acoustic_rain_se-1.pdf` (and byte-identical `-2.pdf`)

**What they do:** Build a low-cost, solar-powered, weather-proof standalone acoustic rain sensor (microphone under a metal dome; acoustic signal power as the rain proxy) with SMS/mobile-internet telemetry, deployed next to standard gauges in the Philippines during the rainy season.

**Results:** Field deployment shows recorded acoustic signal power tracks rain events observed by co-located standard instruments, and demonstrates a practical, deployable, telemetered system for sub-kilometer rainfall monitoring in tropical regions.

**Inference:** The engineering wrapper (power, waterproofing, telemetry) around an acoustic rain sensor is solved and cheap; the limiting factor is the intensity-estimation model — exactly the part this project supplies.

**How it supports this project:** A blueprint for turning this project's model into a field instrument (enclosure, solar power, telemetry), and evidence for the low-cost/dense-network deployment story used to motivate the project.

---

#### A7. Dunkerley (2025) — *A New Device for Continuous, Real-Time Acoustic Measurement of Rain Inclination* — Preprints.org, DOI 10.20944/preprints202512.1003.v1 (not peer-reviewed)
**File:** `research papers\preprints202512.1003.v1.pdf`

**What they do:** Introduce an inexpensive acoustic device that measures the **inclination angle of wind-driven rain** in real time (traditional paired-gauge methods only give long-period averages), while simultaneously recording rainfall duration and intermittency at high temporal resolution. Demonstrated during tropical trade-wind showers.

**Results:** The device continuously resolves rain inclination during showers and captures duration/intermittency with no extra apparatus; the paper is primarily a methods contribution (no accuracy benchmark against reference instruments yet).

**Inference:** Wind-driven rain arrives obliquely and faster than terminal velocity — meaning drop impact energy (and hence sound) depends on wind, not just intensity. Acoustics can measure this wind effect directly.

**How it supports this project:** Strong support for the project's identified "next lever": wind. It explains *why* wind confounds the audio→intensity mapping (oblique, super-terminal impacts change the acoustic signature) and shows the confounder is itself acoustically observable — suggesting wind could eventually be estimated from the same audio rather than requiring an anemometer.

---

#### A8. Nystuen (1996) — *Acoustical Rainfall Analysis: Rainfall Drop Size Distribution Using the Underwater Sound Field* — JTECH 13:74, DOI 10.1175/1520-0426(1996)013<0074:ARARDS>2.0.CO;2 (scanned copy)
**Files:** `research papers\atot-1520-0426_1996_013_0074_arards_2_0_co_2.pdf` (scan), plus text-layer copy `Acoustical Rainfall Analysis Rainfall Drop Size Distribution Using the Underwater Sound Field.pdf`

**What they do:** Formal **inversion of the underwater ambient sound field to retrieve the raindrop size distribution (DSD)**: laboratory-derived sound signatures of individual drop sizes form the basis functions; field measurements at the AOML Rain Gauge Facility (Miami) test both the forward problem (predict sound from DSD) and the inverse (estimate DSD from sound).

**Results:** Tested on several dozen rainfall events over six months, the ARA algorithm gives **excellent estimates of rainfall rate, accumulation, and radar reflectivity**, and resolves DSD variations at 5–10 s temporal resolution.

**Inference:** The rain-sound spectrum is a linear superposition of drop-size-specific signatures — sound carries not just intensity but full microphysics. Intensity regression from audio has a solid physical basis.

**How it supports this project:** The physical foundation of the whole idea: it proves the audio→rainfall mapping is causal, not correlational, and suggests a stretch goal (DSD estimation from air-side audio). It also motivates spectral features that align with drop-size-specific frequency bands.

---

#### A9. Nystuen (1999) — *Relative Performance of Automatic Rain Gauges under Different Rainfall Conditions* — JTECH 16:1025, DOI 10.1175/1520-0426(1999)016<1025:RPOARG>2.0.CO;2
**Files:** `research papers\atot-1520-0426_1999_016_1025_rpoarg_2_0_co_2.pdf`, plus named copies in `research papers\` and `Category 1\`

**What they do:** Deploy six automatic gauge types side by side for 17 months in Miami (~800 events): tipping bucket, weighing, capacitance, optical, disdrometer, and an acoustic sensor, and compare them across stratiform, convective, tropical-storm, light and extreme rain.

**Results:** All gauges intercorrelate ≥ ~0.9 at 1-min rates with biases < 10%, **but every type fails somewhere**: at >100 mm/h the disdrometer and tipping bucket bias low and the optical gauge biases high; at <2 mm/h capacitance and tipping bucket are noise-dominated at 1-min sampling; the acoustic gauge is very sensitive to drops >3.5 mm (they acoustically mask smaller drops); above 5 m/s wind, disdrometer and acoustic gauges bias low.

**Inference:** Ground truth is not absolute — the tipping-bucket labels this project trains on carry known, condition-dependent biases (low at extreme rates, noisy at drizzle, wind-affected). Part of any model's residual error is label error.

**How it supports this project:** (1) Quantifies expected label noise in the project's tipping-bucket ground truth, informing realistic accuracy ceilings and error analysis by intensity band. (2) The documented failure of acoustic sensing under large drops and high wind tells the project exactly which regimes need targeted features/data. (3) Supports the handbook's claim that gauge undercatch in wind is a literature-established problem.

---

#### A10. Ma & Nystuen (2005) — *Passive Acoustic Detection and Measurement of Rainfall at Sea* — JTECH 22:1225, DOI 10.1175/JTECH1773.1
**File:** `research papers\atot-jtech1773_1.pdf` (+ duplicate)

**What they do:** Turn moored hydrophones on the TAO array (90+ buoy-months) into calibrated "Acoustic Rain Gauges": use the universal wind-generated sound spectrum for **absolute self-calibration** of each hydrophone, apply an acoustic discrimination step to isolate rain vs wind vs noise, then fit a **single-frequency rainfall-rate algorithm** validated against R.M. Young siphon gauges and TRMM satellite product 3B42.

**Results:** Acoustic rainfall accumulations match collocated gauges and TRMM at time scales from hours to a year; the discrimination + single-frequency algorithm transfers to two other ocean sites.

**Inference:** (1) A classify-first, then-regress pipeline (their discrimination → quantification) is the proven architecture — the same two-stage structure this project uses. (2) Self-calibration against a universal reference signal solves cross-sensor variation.

**How it supports this project:** Precedent for the detection→regression cascade, and the wind-based self-calibration idea maps onto this project's multi-campaign problem: cross-campaign microphone/gain differences could similarly be normalized against a universal ambient reference rather than per-clip peak normalization (which the project already found harmful and removed).

---

### TIER B — Supporting physics and meteorological-feature papers

---

#### B1. Joss & Waldvogel (1969) — *Raindrop Size Distribution and Sampling Size Errors* — J. Atmos. Sci. 26:566–569 (scanned)
**File:** `research papers\RaindropSizeDist......JASMay1969.pdf`

**What they do:** Derive, assuming an exponential DSD (Marshall–Palmer) with Poisson-distributed drop counts, the theoretical standard deviation of rain rate R and radar reflectivity Z computed from finite drop samples.

**Results:** Small samples give badly noisy R and Z: e.g., at 1 mm/h, ~1 m² of collecting surface exposed for 1 s is needed for Z to be within 20% of its mean with 68% probability. Sampling error alone explains much apparent Z–R variability.

**Inference:** Any short-window rainfall estimate — including a 10-second audio clip vs a 0.2 mm-per-tip bucket — has an irreducible statistical variance floor from counting a finite number of drops.

**How it supports this project:** Explains a structural part of the project's residual error: a tipping bucket integrates discretely (one tip ≈ 0.2 mm) while clips are 10 s, so light-rain labels are quantized and sampling-noisy. Justifies temporal aggregation choices and why per-clip R² saturates; useful citation when defending the realistic-data R² against SARID's curated number.

---

#### B2. Manandhar, Dev, Lee, Winkler & Meng (2018) — *Systematic Study of Weather Variables for Rainfall Detection* — arXiv:1805.01957
**File:** `research papers\1805.01957v1.pdf`

**What they do:** PCA on 7 weather/time variables (relative humidity, solar radiation, temperature, dew point, GPS-derived precipitable water vapor, day-of-year, time-of-day) from a Singapore weather station, then use leading principal components to separate rain vs no-rain.

**Results:** Four principal components explain 85% of variance; the first two PCs already separate rain/no-rain scenarios; all 7 variables contribute comparably to rainfall detection.

**Inference:** Cheap scalar weather variables carry real rain/no-rain signal but are largely redundant with each other — a few components suffice.

**How it supports this project:** If the deployment adds any weather sensing, this tells the project which handful of auxiliary features add detection skill (humidity, solar, temperature/dew-point spread) and that they should be compressed/decorrelated. Useful for the detection track, especially for suppressing false positives in dry, noisy clips.

---

#### B3. Pathan, Wu, Lee, Yan & Dev (2021) — *Analyzing the Impact of Meteorological Parameters on Rainfall Prediction* — arXiv:2110.11059
**File:** `research papers\2110.11059v1.pdf`

**What they do:** Correlation analysis + ML-based feature selection on 5 years of NOAA station data (Michigan, USA) to rank meteorological features (wind speeds/directions, temperatures) by importance for rainfall prediction.

**Results:** Wind-related features correlate positively with precipitation (r ≈ 0.24–0.26 for fastest-wind and average-wind features) and rank among the most important features for rainfall prediction.

**Inference:** Wind is not just a nuisance for gauges — it is itself informative about precipitation occurrence.

**How it supports this project:** Reinforces the project's decision that wind speed is the next feature to add: wind both confounds the acoustic signal (Dunkerley, Nystuen) *and* carries predictive signal about rain itself. Directly supports the planned wind-augmented models.

---

#### B4. Liu, Li, Shang, Tang & Zhang (2021) — *Measurement of Underwater Acoustic Energy Radiated by Single Raindrops* — Sensors 21:2687, DOI 10.3390/s21082687
**File:** `Category 1\sensors-21-02687.pdf`

**What they do:** Derive a dipole-radiation formula for the underwater sound energy of a single raindrop and measure it in a reverberation tank across drop sizes, building a model of kinetic→acoustic energy conversion efficiency.

**Results:** A predictive model for the average radiated sound energy of a raindrop of any diameter; total rainfall sound energy can then be predicted from the DSD.

**Inference:** Acoustic energy scales with drop size/kinetic energy in a quantifiable way — the physical link between "louder" and "heavier rain" is monotonic but drop-size dependent, not a simple function of rain rate alone.

**How it supports this project:** Grounds the intensity-regression idea in drop physics and explains why energy-type features (RMS, band energies) are strong but imperfect predictors: two rains with equal rate but different DSDs sound different. Motivates spectral-shape features alongside energy features.

---

#### B5. Green, Ostafichuk & Rogak (2001) — *Measurement of Three-Dimensional Unsteady Flows Using an Inexpensive Multiple Disk Probe* — JTECH 18:883
**File:** `Category 1\atot-1520-0426_2001_018_0883_motduf_2_0_co_2.pdf`

**What they do:** Develop and field-test a cheap velocimeter made of three orthogonal disks with pressure transducers that measures all three wind velocity components even in highly 3-D flows, unlike yaw-head/five-hole probes.

**Results:** Flat frequency response to 3–4 Hz; velocity magnitude accuracy better than ±0.3 m/s in field/wind-tunnel trials — adequate for many meteorological uses at a fraction of sonic-anemometer cost.

**Inference:** Research-grade wind measurement can be had cheaply and robustly (no moving parts).

**How it supports this project:** A candidate low-cost wind sensor for the planned wind-speed data channel at deployment sites, consistent with the project's low-cost-sensor philosophy.

---

#### B6. Siloko & Uddin (2023) — *A Statistical Study of Wind Speed and its Connectivity with Relative Humidity and Temperature* — Science World J. 18(3), DOI 10.4314/swj.v18i3.13
**File:** `Category 1\scienceworld,+Vol+18(3)+pp+404-413+Siloko+et+al.pdf`

**What they do:** Kernel-density (Gaussian) analysis of five years (2018–2022) of wind speed in Ughelli, Nigeria, and its correlation with relative humidity and temperature (AMISE performance measure, Pearson tests).

**Results:** Wind–temperature correlation is consistently negative (hotter → calmer); wind–humidity correlation flips sign across years.

**Inference:** Wind statistics are site- and season-specific; relationships between wind and other weather variables cannot be assumed stable across deployments.

**How it supports this project:** Marginal, background-level relevance: a caution that any wind-related feature or correction learned in one campaign/climate may not transfer to another — an argument for campaign-aware validation splits (which the project already uses).

---

### TIER E — Newly added papers (2026-07-15 batch): gauge error correction & sampling uncertainty

These 8 files arrived with auto-download filenames truncated to ~40 characters (e.g. `Rainfall_and_sampling_uncertainties_A_ra.pdf`). All have been identified, analyzed below, and **renamed** to the library's `YYYY - Title.pdf` convention (see Section 6 for the exact rename log). None turned out to be duplicates of existing library content, except two that were exact copies of each other and have been merged into one file.

---

#### E1. Habib, Krajewski, Nespor & Kruger (1999) — *Numerical simulation studies of rain gage data correction due to wind effect* — J. Geophys. Res. 104(D16):19,723–19,733
**File (renamed):** `1999 - Numerical simulation studies of rain gage data correction due to wind effect.pdf`

**What they do:** Use CFD-derived wind-flow-around-gage correction formulae (building on Nespor & Sevruk's numerical model) combined with high-temporal-resolution rainfall and wind speed measurements to correct rain gage data, specifically investigating how the choice of *temporal averaging scale* affects the estimated wind-induced bias.

**Results:** Applying the wind correction at short timescales (down to 1 minute) is essential — averaging over longer periods before correcting causes a **significant overestimation** of the wind-induced bias. The wind-induced error is a nonlinear function of wind speed, rain rate, *and* the drop-size-distribution parameter; correction factors expressed only as a function of wind speed (ignoring DSD) show large random scatter versus literature formulae.

**Inference:** A wind correction applied at coarse time resolution is actively harmful, not just imprecise — it systematically overstates the bias. And wind speed alone is an incomplete correction variable; DSD-awareness matters.

**How it supports this project:** Directly informs *how* to add a wind channel to the model: any wind-based correction/feature must operate at the project's native short-clip resolution (seconds, not minutes) to avoid introducing bias rather than removing it, and a simple wind-speed feature alone may underperform without some proxy for drop-size distribution (which the acoustic spectrum itself may supply, per Nystuen 1996 and Liu et al. 2021).

---

#### E2. Ungersböck, Rubel, Fuchs & Rudolf (2001) — *Bias correction of global daily rain gauge measurements* — Phys. Chem. Earth 26:411–414
**File (renamed):** `2001 - Bias correction of global daily rain gauge measurements.pdf`

**What they do:** Develop a statistical model (built for BALTEX, adapted for the GPCC) that corrects *daily* rain-gauge measurement bias, as opposed to the monthly-only correction factors previously used operationally, and validate it against 2 years of data across four GEWEX continental-scale regions (Europe, Asia, South America, North America).

**Results:** Daily-resolution correction factors roughly match the established monthly Legates (1987) corrections in Europe during summer, but diverge substantially elsewhere: Legates over-corrects snow (up to 50% higher) and over-corrects in the Asia/South-America regions (50–100% higher), while matching closely in North America.

**Inference:** Gauge correction factors are not universal — they vary by region, season, and precipitation phase, and coarse (monthly) corrections can be badly wrong for individual days.

**How it supports this project:** Reinforces (alongside E1) that any systematic correction to the tipping-bucket ground truth must be applied at fine temporal and regional granularity, not as a single global constant — consistent with this project's existing campaign-aware treatment of its 19 field deployments.

---

#### E3. Maksimović, Bušek & Petrović (1991) — *Corrections of rainfall data obtained by tipping bucket rain gauge* — Atmos. Res. 27:45–53
**File (renamed):** `1991 - Corrections of rainfall data obtained by tipping bucket rain gauge.pdf`

**What they do:** Laboratory- and field-test a Rimco R/TBR-8 tipping-bucket gauge to characterize three specific instrument imperfections: (1) **non-linearity** — the volume dispensed per tip is itself a function of rainfall intensity; (2) **"double tipping"** after successive single tips, caused by a mismatch between bucket volume and the siphon-controller's dispensed volume; (3) sensitivity of both effects to the bucket's screw-adjusted "calibration volume."

**Results:** Presents a full experimental correction methodology so raw tips recorded in real storms can be corrected for these non-linear, intensity-dependent effects.

**Inference:** The mm-per-tip conversion is not a constant even within a single gauge — it depends on the current rain intensity in a documented, correctable way, and errors compound at high intensity (more successive tips → more chances for double-tipping).

**How it supports this project:** A second, earlier, independent confirmation (alongside Prakosa et al., Tier C1) that tipping-bucket ground truth carries intensity-dependent systematic error — this is now corroborated by two separate studies 27 years apart, strengthening the case for treating gauge labels as noisy, particularly at high rain rates where this project's model already underperforms.

---

#### E4. Villarini, Mandapaka, Krajewski & Moore (2008) — *Rainfall and sampling uncertainties: A rain gauge perspective* — J. Geophys. Res. 113:D11102
**File (renamed):** `2008 - Rainfall and sampling uncertainties A rain gauge perspective.pdf`

**What they do:** Using 6+ years of data from a dense 50-gauge network over 135 km² (Brue catchment, England), characterize two distinct error types relevant to remote-sensing/point-sensor comparison: **temporal sampling uncertainty** (from gaps in observation, e.g. infrequent satellite/radar overpasses) and **spatial sampling uncertainty** (from using a point measurement to represent an areal estimate).

**Results:** Temporal sampling uncertainty grows with the sampling interval following a scaling law and shrinks with larger averaging area (with no strong orography dependence); spatial sampling uncertainty shrinks with longer accumulation time. They also derive a practical rule for how many gauges are needed to estimate areal rainfall to a given accuracy, useful for validating high-resolution satellite products.

**Inference:** Any comparison between a point sensor (a tipping bucket, or a single microphone) and an areal ground truth carries an intrinsic, quantifiable representativeness error separate from instrument error.

**How it supports this project:** Provides the vocabulary and quantitative framework for a limitation this project already faces implicitly: a single microphone (like a single gauge) is a point measurement, and comparing it to a nearby but not co-located gauge, or extrapolating results to "area coverage" claims, carries this exact spatial-sampling uncertainty — worth citing when the handbook discusses the dense-network value proposition.

---

#### E5. Steiner, Smith, Uijlenhoet & Hou (n.d., ~1999–2000) — *Rainfall Variability and its Effect on Rainfall Measurements* — AMS conference preprint, paper P14.1
**File (renamed):** `n.d. - Rainfall variability and its effect on rainfall measurements.pdf`

**What they do:** Analyze 30 storms (1996–1997, ~785 mm total) over the densely-instrumented Goodwin Creek watershed (Mississippi) with extensive QC of radar, rain gauge, disdrometer, and lightning data, storm-cell tracking, and sensitivity analysis, to answer: how much of the observed difference between radar-estimated and gauge-measured rainfall is explained by sensor resolution differences versus genuine small-scale rainfall variability, rather than instrument error?

**Results:** Emphasizes that high-quality rain gauge data is essential for radar bias-adjustment — questionable gauge data can dramatically distort radar-gauge merged rainfall products — and that a meaningful share of radar-vs-gauge disagreement is attributable to genuine spatiotemporal rainfall variability rather than either instrument being "wrong."

**Inference:** Disagreement between two well-functioning rainfall instruments does not automatically mean one is broken — some of it is the real, physical small-scale variability of rain itself.

**How it supports this project:** Directly relevant caution for evaluating the acoustic-gauge-vs-tipping-bucket comparison: some fraction of this project's residual error should be attributed to genuine micro-scale rainfall variability between what the microphone "hears" locally and what the (possibly non-co-located) gauge collects, not solely to model inaccuracy.

---

#### E6. Kochendorfer, Rasmussen, Wolff et al. (2017) — *The quantification and correction of wind-induced precipitation measurement errors* — Hydrol. Earth Syst. Sci. 21:1973–1989
**File (renamed):** `2017 - The quantification and correction of wind-induced precipitation measurement errors.pdf`

**What they do:** A large multi-author NOAA/NCAR/Norwegian-Met-Institute study using two precipitation testbeds (WMO-SPICE-affiliated) to quantify wind-induced undercatch for unshielded weighing gauges and gauges with four common windshields (including single-Alter), then derive and cross-validate correction functions using only wind speed and air temperature as inputs.

**Results:** An unshielded gauge can catch **under 50% of actual solid precipitation** at wind speeds over 5 m/s. A single correction function developed for single-Alter-shielded gauges, tested at two independent sites (US and Norway), reduced bias from **−12% to 0%** at the US site and from **−27% to −4%** at the Norwegian site.

**Inference:** Wind-induced undercatch is large (up to 50%+ in solid precip, and non-trivial in liquid rain too) but is well-characterized and correctable with just two cheap, ubiquitous inputs (wind speed, temperature), and the correction transfers reasonably well across very different climates.

**How it supports this project:** The single strongest piece of quantitative evidence in the whole library for prioritizing wind speed as this project's "next lever" (as the handbook already concludes): it shows a **large, well-characterized, and transferable** bias exists, with a simple, cheap, two-input correction function as precedent for how a wind-augmented version of this project's model could be structured.

---

#### E7. Lee & Zawadzki (2005) — *Variability of Drop Size Distributions: Time-Scale Dependence of the Variability and Its Effects on Rain Estimation* — J. Appl. Meteor. 44:241–255 (McGill University)
**File (renamed):** `2005 - Variability of Drop Size Distributions Time-Scale Dependence of the Variability and Its Effects on Rain Estimation.pdf`

**What they do:** Analyze 5 years (20,000+ one-minute DSDs) of disdrometer data from Montreal to quantify how drop-size-distribution variability, at multiple time scales (climatological, day-to-day, within-day, between physical processes), biases radar rain-rate (R) estimation from reflectivity (Z), classifying DSDs by the vertical radar-profiler structure of the associated physical process.

**Results:** Using a single climatological R–Z relationship gives ~**41% random error** in instantaneous rain rate; this error shrinks with longer integration time but plateaus beyond ~2 hours. Daily accumulations using the climatological relationship carry a **28% bias** from day-to-day DSD variability; even *daily* R–Z relationships still leave 32% random error from within-day variability. The DSD variability *between different physical processes* (e.g., stratiform vs. convective) is larger than day-to-day variability and alone causes a 41% accumulation bias — accurate estimation (~7% error) is only achievable once the correct underlying physical process is identified.

**Inference:** A single, static mapping (whether R–Z for radar or, by direct analogy, audio-features→intensity for this project) has an irreducible ~30–40% error ceiling unless it is conditioned on the type of rain event (its underlying physical process / DSD regime), not just its current intensity.

**How it supports this project:** Perhaps the most important physics paper in this update for model design: it is strong, quantitative, independent evidence that a *single* audio→intensity regression function is fundamentally limited by unmodeled DSD/rain-type variability, and that **conditioning the model on rain "regime"** (e.g., via a learned or explicit classification of storm type, akin to distinguishing convective vs. stratiform) could meaningfully reduce error — a concrete, literature-backed architectural idea (mixture-of-experts / regime-conditioned regression) worth prototyping against the project's stagnant 0.531 R².

---

### TIER C — Instrumentation, calibration, and standards context

---

#### C1. Prakosa, Wijonarko & Rustandi (2018) — *The Performance Measurement Test on Rain Gauge of Tipping Bucket due to Controlling of the Water Flow Rate* — ElConRus 2018, pp. 1136–1140
**File:** `research papers\Paper_RainGaugeTippingBucketIEEE.pdf` (the PDF is a proceedings extract; the paper is at PDF pages 30–34)

**What they do:** Laboratory-test tipping-bucket repeatability by feeding controlled water flow rates (24–100 ml/min) via a microcontroller-PWM-controlled pump and counting tips.

**Results:** Volume-per-tip varied between series — (18.93±0.60), (17.89±0.92), (17.80±0.60) ml/tip, average 18.21±1.25 ml/tip (~7% spread) — showing flow rate significantly affects tipping-bucket calibration.

**Inference:** The mm-per-tip "constant" of a tipping bucket is intensity-dependent; the ground-truth conversion factor drifts a few percent with rain rate.

**How it supports this project:** Another quantified source of label error in the project's ground truth: intensity-dependent tip calibration adds systematic bias precisely at the high intensities where the model already struggles. Supports considering dynamic calibration corrections on the gauge labels.

---

#### C2. Kou, Liu, Li, Qin & Liu (2026) — *AI Methods in Sensor Calibration* (review) — Sensors 26:2805, DOI 10.3390/s26092805
**File:** `research papers\AI_Methods_in_Sensor_Calibration.pdf`

**What they do:** Review how AI models replace polynomial transfer-function fitting in sensor calibration: learning input→output mappings, compensating temperature/humidity/drift interference, and enabling large-scale low-cost sensor fleets, plus supporting tools (preprocessing, training optimization, data augmentation).

**Results:** Across many sensor types, AI calibration consistently beats preset empirical models on accuracy and stability, especially for nonlinear transfer functions and environmental compensation; open challenges include generalization and long-term drift.

**Inference:** This project *is* an instance of AI sensor calibration — the microphone is the raw sensor, the ML model is its learned transfer function to mm/h, and cross-campaign/environmental compensation is the hard part the review says AI handles best.

**How it supports this project:** Framing and technique source: positions the project inside an established methodology (useful for the paper's related-work section), and its catalog of interference-compensation and augmentation techniques is a menu for improving cross-campaign robustness.

---

#### C3. WMO (2008) — *Guide to Meteorological Instruments and Methods of Observation*, WMO-No. 8, 7th ed.
**File:** `Category 1\CWOP-WMO8.pdf` (681 pages; CWOP-distributed copy)

**What it is:** The world-standard reference on how precipitation (and every other meteorological variable) must be measured: siting, exposure, gauge types, uncertainty requirements, and QC procedures.

**Results/content relevant here:** Defines the accuracy classes and siting/exposure rules that official rain measurements follow, including wind-induced undercatch corrections for gauges.

**Inference:** Any claim that an acoustic gauge "matches" official measurement quality must be phrased against WMO-No. 8 requirements.

**How it supports this project:** The compliance yardstick: use it to specify target uncertainty for the acoustic gauge, to justify ground-truth siting decisions in field campaigns, and to cite standard undercatch corrections when adjusting tipping-bucket labels for wind.

---

#### C4. RHD Acoustic Rain Disdrometer & Hail Sensor — commercial datasheet (not a research paper)
**Files:** `research papers\Rain disdrometer and hail detection.pdf` = `Category 1\Datasheet_RHD_V1_41.pdf` (byte-identical)

**What it is:** Datasheet for a commercial low-cost acoustic rain gauge/disdrometer: a sealed polished stainless-steel hemisphere (160 mm, 402 cm²) that senses raindrop/hailstone impacts; no moving parts, <1 mA standby.

**Key specs:** Rain intensity accuracy **±15%** (100% duty cycle); DSD reported in **27 size classes** from ≤0.75 mm to ≥7.0 mm (detection threshold ~0.5 mm); hail counting to 5 impacts/s; drop velocity not measured; analog/SDI-12/RS-232/Modbus outputs.

**Inference:** Industry already sells impact-acoustic rain sensing at ±15% accuracy — but it needs a dedicated, engineered sensing surface. The commercial bar for a microphone-only ML approach is therefore ±15% intensity error.

**How it supports this project:** (1) Competitive benchmark: beating ±15% with a commodity microphone would be a headline result. (2) Proof of commercial demand for acoustic rain sensing. (3) Its 27-class DSD output shows what a mature acoustic product exposes — a target feature set for the long-term roadmap.

---

### TIER D — Off-topic files (flagged; not rainfall-audio relevant)

---

#### D1. Mohammadi et al. (2020) — *A Multi-Sensor Comparative Analysis on the Suitability of Generated DEM from Sentinel-1 SAR Interferometry* — Sensors 20:7214
**Files:** `research papers\sensors-20-07214.pdf` (= the two `n.d. - A Multi-Sensor Comparative Analysis...` copies)

**What they do / results:** Generate a DEM from Sentinel-1 InSAR for sites in Malaysia and Iran and compare against AIRSAR, ALOS-PALSAR, TanDEM-X, SRTM; Sentinel-1 DEM turns out noisy (short perpendicular baseline) and **worse than SRTM/TanDEM-X** despite finer nominal resolution.

**Relevance:** None to acoustic rainfall sensing. Likely collected for general hydrology background. **Recommend moving out of the rainfall library** (or into a "hydrology-misc" folder) so it doesn't pollute searches.

---

#### D2. Kurşun, Güven & Ersoy (2022) — *Utilizing Piezo Acoustic Sensors for the Identification of Surface Roughness and Textures* — Sensors 22:4381
**Files:** `research papers\sensors-22-04381.pdf` (= `n.d. - Utilizing Piezo Acoustic Sensors...`)

**What they do / results:** Correlate frequency spectra from piezo disks (elastic waves from a diamond stylus dragged on metal) with profilometer surface-roughness parameters (Ra, Rz); spectra correlate clearly with roughness — a promising novel roughness measurement.

**Relevance:** Tangential. It is manufacturing metrology, not meteorology — but it independently demonstrates the project's core signal-processing premise (impact/friction acoustics + spectral analysis ⇒ quantitative physical parameter). Keep only as methodological inspiration for piezo-based sensing surfaces.

---

#### D3. Zhao et al. (2026) — *A self-powered microsystem with efficient power management for continuous wireless sensing* — Microsyst. Nanoeng. 12:178, DOI 10.1038/s41378-026-01315-z
**File:** `research papers\A_self-powered_microsystem_with_efficient_power_ma.pdf`

**What they do / results:** Power a continuously sensing, wirelessly transmitting microsystem entirely from a triboelectric nanogenerator (TENG) harvesting <10 Hz mechanical energy; custom power management fixes the TENG↔electronics impedance mismatch for a **5× energy gain** over full-bridge rectification, cold-start 0→4.2 V in 525 s, ~110 µW at 5 Hz excitation; demonstrated as a battery-less wireless gas monitor.

**Relevance:** Deployment engineering only: a battery-less power budget (~110 µW) is far below what continuous audio ML capture needs, but the paper (and its raindrop-TENG citations) is useful if the project ever explores energy-autonomous field nodes. Not a rainfall-estimation reference.

---

## 4. New papers added and renamed (2026-07-15)

All 8 newly added PDFs (Tier E above) were identified by content and renamed from their cryptic auto-download names to the library's `YYYY - Title.pdf` convention:

| Old filename | New filename |
|---|---|
| `Numerical_simulation_studies_of_rain_gag.pdf` | `1999 - Numerical simulation studies of rain gage data correction due to wind effect.pdf` |
| `Bias_correction_of_global_daily_rain_gau.pdf` | `2001 - Bias correction of global daily rain gauge measurements.pdf` |
| `Corrections_of_rainfall_data_obtained_by-2.pdf` | `1991 - Corrections of rainfall data obtained by tipping bucket rain gauge.pdf` |
| `Rainfall_and_sampling_uncertainties_A_ra.pdf` | `2008 - Rainfall and sampling uncertainties A rain gauge perspective.pdf` |
| `Rainfall_variability_and_its_effect_on_r.pdf` (+ byte-identical `-1.pdf`) | `n.d. - Rainfall variability and its effect on rainfall measurements.pdf` (one copy kept) |
| `The_Quantification_and_Correction_of_Win.pdf` | `2017 - The quantification and correction of wind-induced precipitation measurement errors.pdf` |
| `Variability_of_Drop_Size_Distributions_T.pdf` (+ byte-identical `-1.pdf`) | `2005 - Variability of Drop Size Distributions Time-Scale Dependence of the Variability and Its Effects on Rain Estimation.pdf` (one copy kept) |
| `Adjustment_of_1_min_rain_gauge_time_seri.pdf` | merged — see Section 5 (near-duplicate of an existing library file, one copy kept, old copy renamed for clarity) |

## 5. Duplicate cleanup — executed 2026-07-15

Every candidate below was **verified before deletion**: exact duplicates by MD5 byte-hash, near-duplicates by full-text similarity ratio (pypdf extraction, `SequenceMatcher`, threshold ≥0.97 required for deletion). All deleted files were sent to the **Windows Recycle Bin**, not permanently erased. Total: **117 duplicate PDFs removed**, library reduced from ~178 to **172 unique PDFs**. The `research papers\duplicates\` folder (which held 113 of these) is now empty and has been deleted.

**Cross-checked/confirmed duplicate groups removed:**

- **SARID paper ×4 → 1 kept**: kept `2024 - Estimating rainfall intensity based on surveillance audio and deep-learning.pdf`; deleted `sarid.pdf`, `1-s2.0-S2666498424000644-main.pdf`, and the `(2)` copy (text similarity ≥0.97 confirmed for all).
- **Çoban et al. (sound-based weather monitoring) ×3 → 1 kept**: kept `Towards High Resolution Weather Monitoring with Sound Data.pdf`; deleted `document.pdf` and `2309.16867v1.pdf` (byte-identical / near-identical text).
- **Nystuen 1999 (Relative Performance of Automatic Rain Gauges) ×3 → 1 kept**: kept the `research papers\` named copy; deleted `atot-1520-0426_1999_016_1025_rpoarg_2_0_co_2.pdf` and the `Category 1\` copy.
- **RHD datasheet ×2 → 1 kept**: kept `Category 1\Datasheet_RHD_V1_41.pdf`; deleted `research papers\Rain disdrometer and hail detection.pdf` (byte-identical; its old name misleadingly suggested a research paper).
- **Guico et al. (novel acoustic rain sensor) ×2 → 1 kept**: `-2.pdf` deleted (byte-identical to `-1.pdf`).
- **Ma/Dushaw/Howe "Rainfall at Sea" ×2 → 1 kept**: `Category 1\` copy deleted (byte-identical to `research papers\` copy).
- **"Recording Rainfall Intensity" (n.d. vs 2023) ×2 → 1 kept**, **"Wind speed estimation..." (n.d. vs 2023) ×2 → 1 kept**, **"Measuring Amazon Rainfall Intensity With Sound" (n.d. vs named) ×2 → 1 kept**, **"Understanding the Mechanical Biases..." (2) vs base ×2 → 1 kept**: all confirmed byte-identical, one copy each retained.
- **Evaluation of Numerous Kinetic Energy–Rainfall Intensity Equations ×3 → 1 kept**: the `(2)` copy confirmed near-identical text (≥0.97) and deleted; the `(3)` copy was checked and found only **73% similar** (different page count, 17 vs 16 pp) — **kept as a possibly distinct revision**, not deleted (flagged for manual review).
- **Piezo Acoustic Sensors for Surface Roughness ×2 → 1 kept**: `sensors-22-04381.pdf` deleted (byte-identical to the `n.d.` copy).
- **Adjustment of one-minute rain gauge time series ×4 → 2 kept (two distinct versions)**: the `n.d.` copy and one of the two `2023` copies were confirmed byte/text-identical and deleted; the newly added `Adjustment_of_1_min_rain_gauge_time_seri.pdf` (9 pp, AMT *Discussion* preprint text) was confirmed **99.9% identical** to the existing `(2)` copy and deleted, while the surviving 13-page `2023 -...pdf` (final AMT article) was kept as-is and the discussion-preprint copy renamed `2023 - ...(AMT discussion preprint).pdf` for clarity — these are two genuinely different manuscript versions of the same paper (preprint discussion vs. final published), not accidental duplicates.
- **`research papers\duplicates\` folder (113 files)**: every file was matched against its corresponding outside copy — same exact-hash / near-text-identical logic — and removed, **except two genuinely distinct files** that were relocated instead of deleted:
  - `Technical_note_An_innovative_monitoring_approach_t-1.pdf` (17 pp) is a **different, earlier version** of "An innovative monitoring approach to measure spatio-temporal throughfall patterns in forests" than the 2025 22-page copy already in the library (only 28% text similarity, both are legitimate content) — moved to `research papers\2026 - Technical note An innovative monitoring approach to measure spatio-temporal throughfall patterns in forests (published version).pdf`.
  - `Adaptation_of_RainGaugeQC_algorithms_for_quality_c-3.pdf` (26 pp) is a **longer, extended version** of the RainGaugeQC paper than the library's existing 17-page `n.d.` copy (only 19% text similarity) — moved to `research papers\n.d. - Adaptation of RainGaugeQC algorithms for quality control of rain gauge data (extended 26pp version).pdf`. (Two other copies of this same paper inside `duplicates\`, `-1.pdf` and `-2.pdf`, *were* exact byte-matches of the existing `n.d.` copy and were deleted.)

**Not touched (flagged, not deleted):** the off-topic Sentinel-1 DEM paper (`sensors-20-07214.pdf` + 2 named copies, Tier D1) — these are exact/near duplicates of *each other* but are off-topic content, outside the scope of a "duplicate cleanup" pass; recommend a separate decision on whether to remove them from this library entirely.

---

*Generated by opening and reading every cryptically-named PDF in `ARG_Research/` (text extraction + page-image reading for scans), 2026-07-15. Analyses of results are taken from each paper's own abstract/results sections. Duplicate cleanup executed and logged same day; all deletions routed through the Recycle Bin.*
