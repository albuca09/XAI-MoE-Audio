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

### Option B) Future PyPI release

    pip install xai-moe-audio

---

## Dataset Format

Minimal expected structure:

    dataset/
      train/
        *.wav
      val/
        *.wav
      test/
        *.wav
      metadata.csv

Minimal `metadata.csv` format:

    filepath,split,label
    train/a.wav,train,car
    train/b.wav,train,dog
    val/c.wav,val,dog
    test/d.wav,test,car

Notes:
- `filepath` is relative to dataset root
- `split` is one of: `train`, `val`, `test`
- `label` can be a string name (auto-mapped to ids)

---

## Quickstart (Python API)

### Train

    from xaimoe_audio import MoEAudioClassifier

    clf = MoEAudioClassifier(
        task="single_label",       # or "multi_label"
        budget="accuracy",         # or "latency"
        auto_config=True,          # profiler + policy engine
        explainability="gradcam",  # optional human explanation
        device="cuda",
    )

    clf.fit(
        dataset_dir="dataset",
        metadata_csv="dataset/metadata.csv",
        epochs=20,
        batch_size=16,
        out_dir="runs/exp1",
    )

    clf.print_architecture()
    clf.print_expert_usage()

### Evaluate

    metrics = clf.evaluate(
        dataset_dir="dataset",
        metadata_csv="dataset/metadata.csv",
        split="test",
    )
    print(metrics)

### Inference (single file)

    pred = clf.predict("dataset/test/d.wav")
    print(pred)

### Explainability (Grad-CAM)

    out = clf.explain(
        audio_path="dataset/test/d.wav",
        method="gradcam",
        save_dir="runs/exp1/explain",
    )
    print(out)

---

## CLI Usage

After install:

Train:

    xaimoe-audio train --dataset dataset --metadata dataset/metadata.csv --epochs 20 --batch-size 16 --out runs/exp1

Evaluate:

    xaimoe-audio eval --dataset dataset --metadata dataset/metadata.csv --split test --ckpt runs/exp1/best.pt

Infer:

    xaimoe-audio infer --audio dataset/test/d.wav --ckpt runs/exp1/best.pt

Explain:

    xaimoe-audio explain --audio dataset/test/d.wav --ckpt runs/exp1/best.pt --out runs/exp1/explain

---

## What the user needs to provide (INPUT)

Minimum required:
- `dataset_dir`
- `metadata.csv`
- `task`: `single_label` or `multi_label`

Recommended:
- `budget`: `latency` or `accuracy`
- `device`: `cuda` preferred

Optional:
- fixed target sampling rate
- fixed frequency range
- manual spectrogram kind (override auto-config)
- manual N experts / Kmax
- manual tiling size/stride

---

## What the library outputs (OUTPUT)

During training, it saves:

1) Model artifacts:
- `best.pt`
- `last.pt`

2) Full reproducible configuration:
- `run_config.json` (labels + policy + dataset profile)
- `profile.json`
- `policy.json`

3) Architecture + routing reports:
- `architecture.txt`
- `expert_usage.json` (expert selection rates, K histogram, load balancing proxy)

4) Metrics:
- `metrics.json`
- optional confusion matrix plots

5) Explainability artifacts:
- Grad-CAM overlays
- router saliency maps
- patch selection grids

---

## How Auto-config Works

The library pipeline:

1) profiles dataset:
- fs_mode / fs_unique
- effective band (fmin_eff / fmax_eff)
- SNR proxy, flatness proxy, transient proxy, clipping proxy

2) computes quality score Q and effective bandwidth

3) selects spectrogram backend and parameters:
- fs-respectful
- (fmin, fmax) compatible with PSD proxy

4) defines:
- number of experts N
- maximum K (Kmax)
- tiling size / stride

5) enables routing:
- patches receive different expert budgets
- low quality spectrogram -> fewer experts (avoid overfitting noise)
- high quality + complex saliency -> more experts

---

## Expert Families (default ExpertBank)

Default ExpertBank includes types such as:
- NoiseRobust_Broadband
- NoiseRobust_LowFreq
- NoiseRobust_HighFreq
- TransientExpert
- HarmonicExpert
- FineFreqExpert
- MidbandGeneral
- HighbandGeneral

Routing masks experts that are incompatible with dataset conditions:
- if `fmax_eff < 2000 Hz` -> disables highband experts
- if `Q` is low -> disables fine-resolution experts
- if transients dominate -> increases transient expert probability

---

## Roadmap

- add PaSST / HTSAT / BEATs backbones (wrappers)
- true top-k dispatch batched (speedup)
- Grad-CAM distillation into a proxy router saliency model
- Optuna-based auto-tuning (K/N/patch sizes)
- FCWT/CQT backends

---

## Citation

If you use this repo in academic work, cite:

    @misc{xaimoeaudio2026,
      title   = {XAI-MoE-Audio: XAI-driven Mixture-of-Experts for audio spectrogram classification},
      author  = {Luis Paulo Guedes},
      year    = {2026},
      note    = {GitHub repository}
    }

---

## License

MIT License

