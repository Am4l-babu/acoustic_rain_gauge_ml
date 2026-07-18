# Roadmap — Acoustic Rain Gauge

_Last updated: 2026-07-16. Derived from [DEEP_RESEARCH_ANALYSIS.md](DEEP_RESEARCH_ANALYSIS.md); status facts from [PROGRESS.md](PROGRESS.md) and [`docs/reports/`](reports/)._

**Where we are (measured):** classifier AUC-ROC **0.887**; regressor **R² = 0.5429** (learned stacking ensemble, [`reports/ensemble_stack_report.json`](reports/ensemble_stack_report.json)); 780,725 clips / 19 campaigns / 8 kHz / 10 s; ESP32-S3 + INMP441 firmware built and streaming.

**Benchmarks:** SARID R² 0.765, Monti & Ntalampiras R² 0.787 — both single-site, curated, rain-only. Xavier et al. hourly R² > 0.85 cross-site with a *much* simpler model. RHD commercial acoustic disdrometer: ±15% intensity.

---

## The strategic bet

The research analysis changed what this project should optimize. Three findings drive everything below:

1. **Our per-clip R² is probably the wrong headline number.** A 10-second clip against a 0.2 mm-per-tip bucket is dominated by label quantization at light rain rates. Xavier et al. got R² 0.62 → 0.85+ from the *same model* by aggregating to hourly. We have never tested this, and it costs a few hours.
2. **A single audio→intensity regressor has a physics-imposed ceiling.** Lee & Zawadzki quantify it: ~41% irreducible error unless the model is conditioned on rain *regime*. Our gap to SARID is partly a dataset-difficulty gap, not a competence gap.
3. **Our real differentiator is not accuracy — it's the dataset.** 19 campaigns over 2.5 years with realistic imbalance. Nobody else in this field can measure cross-deployment generalization. That is the contribution worth making.

So: **stop chasing 0.787 on a curated benchmark. Start measuring what our number means, and publish the thing only we can.**

---

## Phase A — Near-term (weeks). Cheap, high-information, no new data.

Everything here uses data and models that already exist.

| # | Task | Why | Effort | Success = |
|---|---|---|---|---|
| **A1** | **Integration-time scaling analysis** | Xavier: R² 0.62→0.85 from aggregation alone. Our labels are tip-quantized; per-clip R² may be measuring label noise, not model error | ~1 day, no retraining | R² vs {10 s, 1, 5, 15 min, 1, 3 h} curve, per campaign. **Either result is publishable** |
| **A2** | **Leave-one-campaign-out CV** | The only honest cross-deployment number obtainable from our data — and the one nobody else can produce | Compute only | 19-fold LOCO R²/AUC; per-campaign breakdown |
| **A3** | **Physics-band features** | 3-way convergence (Xavier + Monti + our own SHAP) on 0–797 Hz (amount) and 1641–2719 Hz (detection) | ~1 day | 2 Welch-PSD scalars added to the master store; A/B vs current 175 |
| **A4** | **Spectral kurtosis + modulation spectrum** | Separates impulsive rain from stationary wind *within* a band — no energy feature can. Absent from the entire library | ~2 days | Added; A/B on false-positive rate |
| **A5** | **Per-campaign dry-ambient calibration** | Ma & Nystuen self-calibration. We have 666k dry clips as a free per-campaign reference. Strictly better than peak-norm *or* no-norm | ~1 day | Cross-campaign variance drops; LOCO (A2) improves |
| **A6** | **Fix the docs** | Handbook flags README/PROGRESS as behind the code | done in this pass | ✅ |

**A1 and A2 first.** They're the cheapest and they may redefine the target.

---

## Phase B — Medium-term (1–2 months). New structure, new comparisons.

| # | Task | Why | Risk | Success = |
|---|---|---|---|---|
| **B1** | **Run our pipeline on public SARID** | Calibrates our 0.5429 against 0.765/0.787 on *their* data. Turns an apparent shortfall into a quantified difficulty gap | Low | Head-to-head table |
| **B2** | **Regime-conditioned mixture-of-experts** | Lee & Zawadzki: 41%→7% *if* regime is known. Our stack's outsized gain suggests it's already doing this implicitly | Medium | Beat 0.5429; regimes physically interpretable |
| **B3** | **Resolve DL degradation at scale** | Open problem. New hypothesis: it's **campaign distribution shift**, not the learning rate — the pilot is IID-shuffled, the full run isn't | Medium | Shuffled-vs-chronological + single-campaign ablations settle it |
| **B4** | **Foundation-model embeddings (BEATs/CLAP/AST)** | Large pretrained gain *predicted for the classifier*; **predicted to disappoint for the regressor** (semantic embeddings are often level-invariant, and level is our label) | Low | Measure both tasks **separately** — the asymmetry is the finding |
| **B5** | **FFT window sweep** | Monti: 4096 helps — **but at 44.1 kHz.** At our 8 kHz the equivalent *duration* is ~1024, i.e. **shorter** than our current 2048. Naive port is backwards | Low | 1024/2048/4096 measured |
| **B6** | **Onset-latency benchmark** | Flood warning cares about detection latency, not mm (NHESS 2023) | Low | Seconds-to-detect distribution |
| **B7** | **Deploy T1 edge tier** | ESP32-S3 computes band features (A3), sends ~40 B. LoRaWAN-compatible. ~1000× bandwidth cut | Medium | On-device features match server within tolerance |

---

## Phase C — Long-term (3–12 months). Needs new hardware or new data.

| # | Task | Gated on | Why it matters |
|---|---|---|---|
| **C1** | **Anemometer on the next campaign** | **Nothing — start now** | Gates C2/C3. Six papers say wind is essential; we have zero wind data. **The cheapest unblock available** |
| **C2** | **Wind-corrected labels** | C1 | Kochendorfer: −27%→−4% bias from wind+temp alone. Treats wind as *label noise* (which it partly is) rather than a feature — explains Monti's null result |
| **C3** | **Wind estimated from the same audio (multi-task)** | C1 | **Highest-ceiling idea in the analysis.** Single-sensor wind-aware rain sensing. Patentable |
| **C4** | **Standardized sensing surface** | Design work | Our transfer function *is* the surface (the mic is in air, not water — Minnaert doesn't apply to us). RHD proves the engineered-hemisphere approach. Converts a modeling problem into a manufacturing one |
| **C5** | **Two-mic coherence** | Second I2S mic | Wind/rain/interference separation with no ML. S3 has the slot |
| **C6** | **Co-deployed disdrometer, one season** | Budget | The only way to *test* rather than infer the DSD/regime hypothesis. Highest scientific value per unit cost |
| **C7** | **Kinetic energy / erosivity target** | — | KE maps to acoustic energy more directly than mm/h does. **We may be better at KE than at intensity** — and soil-erosion modelers want KE |
| **C8** | **Weak supervision from GPM/reanalysis** | — | Çoban: models beat their own noisy teacher. Scales past gauge-paired campaigns |

---

## High-risk / high-reward

| Idea | Payoff | Why it might fail |
|---|---|---|
| **Wind from audio (C3)** | Single-sensor wind-aware ARG. Patent + paper | Needs wind ground truth; wind may not be separable from rain at the mic |
| **DSD from air-side audio** | Would match a commercial disdrometer with a $10 mic | **Minnaert bubble resonance is underwater-only** — we'd need an empirical basis from a co-deployed disdrometer (C6) |
| **Regime MoE (B2)** | Literature says 41%→7% | Regimes may not be acoustically separable at 10 s |
| **Surface-wetness hysteresis** | Free from timestamps; nobody models it | Effect may be small vs other variance |

---

## Deployable today, no research needed

**Gauge fault detection.** Microphone hears rain + co-located bucket reports zero ⇒ maintenance flag. Uses the classifier we already have (AUC 0.887), at the accuracy it already has, with no new work. It is the one application whose requirements our current system already exceeds — worth shipping while the research continues.

---

## Publication / IP / commercial

**Papers, in order of readiness:**
1. **"How far does an acoustic rain gauge travel?"** (A1+A2+B1) — the first cross-deployment generalization benchmark in acoustic rainfall estimation, at operationally-relevant integration times, calibrated against the public SARID benchmark. **Only we can write this.** Directly answers Monti & Ntalampiras's stated future work.
2. **"Regime-conditioned acoustic rainfall estimation"** (B2) — literature-mandated by Lee & Zawadzki, unattempted by anyone.
3. **Dataset paper** — 780,725 clips, 19 campaigns, 2.5 years, realistic imbalance. The field's datasets are all single-site; this is a resource contribution regardless of model results.

**Patent candidates:** wind-from-audio multi-task estimation (C3); standardized sensing surface + band-feature edge pipeline (C4+B7).

**Commercial:** the bar is RHD's **±15% intensity with a dedicated engineered surface**. Matching that with a commodity MEMS mic is the headline product claim. The near-term wedge is gauge-fault detection and dense low-cost networks (Guico's engineering wrapper is solved; OpenIoT's Pi+LoRaWAN node is over-provisioned for what a $10 MCU at T1 can do).

---

## What we are explicitly *not* doing, and why

| Not doing | Why |
|---|---|
| Chasing 0.787 on SARID | Saturating benchmark, curated single-site data. Our value is elsewhere |
| Full DL on-device (T4) | Our DL doesn't beat the stack, and the stack needs 4 models. The **classical track is the deployment track** |
| Re-trying hurdle models as-is | Failed twice (−0.097, 0.076 vs 0.226). **But note:** they gated on *rain/no-rain*; B2 gates on *regime*. The hurdle result refutes one bad gate, **not** mixture-of-experts |
| Camera fusion | Defeats the cheap/night/privacy premise |
| Federated / meta / continual learning | Solutions to problems we have no evidence of |
| LPC, PLP, chroma | Speech/music features; no rain physics behind them |
| Worrying about 8 kHz | **[Resolved]** All evidence-backed bands are < 3.4 kHz, under our 4 kHz Nyquist. Two papers with 3–6× our bandwidth searched their full range and landed inside ours |

---

*Ranked scoring for every direction: [DEEP_RESEARCH_ANALYSIS.md § 13](DEEP_RESEARCH_ANALYSIS.md#13-ranked-research-directions).*
