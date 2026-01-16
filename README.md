# XAI-MoE-Audio

**XAI-driven Mixture-of-Experts (MoE) for fast and robust audio spectrogram classification**, with automatic dataset profiling and adaptive routing.

This library was designed to solve a recurring problem in audio/sonar classification:

✅ Different datasets require different spectrogram backends (STFT / LogMel / PCEN / FCWT)  
✅ Different sampling rates imply different effective frequency ranges  
✅ Spectrogram quality varies (noise, clipping, transients)  
✅ A single monolithic network often fails to cover the full time-frequency space  
✅ Explainability should not be a “human-only” tool — **XAI should feed the routing policy**

So we implement a **fast XAI router** that *interprets* saliency maps (fast saliency always-on, Grad-CAM optional) to decide:

- **which experts** to activate
- **how many experts (K)** to activate
- compatible experts given **fs / fmin / fmax / quality score**

---

## Key Features

### 1) Automatic Dataset Profiling (no user effort)
On training start we extract:

- sampling rate stats (`fs_mode`, `fs_unique`)
- effective band (`fmin_eff`, `fmax_eff`) using PSD proxy
- duration distribution
- SNR proxy
- transient rate proxy (kurtosis-based)
- spectral flatness proxy
- clipping rate
- dataset size and number of classes

This produces `profile.json`.

---

### 2) Auto Spectrogram Backend Selection
Based on dataset profiling:

- STFT / LogMel
- LogMel+PCEN-like (recommended for noisy datasets)
- (optional plugins later) FCWT / CQT

Outputs `spectro_config.json`.

---

### 3) Fast XAI-driven MoE Routing
Routing is **not** hard-coded:

- an always-on *fast saliency* (Gradient×Input or proxy)
- XAI map → patch importance
- combined with dataset attributes (fs, bandwidth, quality score)
- router selects:
  - **Top experts**
  - **K per sample** (adaptive gating)

Outputs:
- `architecture.txt`
- `expert_usage.json`
- sample traces (optional)

---

### 4) Grad-CAM for Human Explanations (optional)
Grad-CAM overlays can be produced for:

- QA / debugging
- reports
- publication figures

Outputs:
- `gradcam.png`
- `saliency_router.png`
- patch-level routing plots

---

## Installation

### Option A) Install from source (recommended for development)
```bash
git clone https://github.com/<YOUR_USER>/xai-moe-audio.git
cd xai-moe-audio
pip install -e ".[dev]"
