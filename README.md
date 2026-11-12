# Acoustic Anomaly Detection for MSMEs (Tech4Bharat 4.3)

Offline-first, acoustic-based early-warning system for machine breakdowns in small workshops and mills.

---

## The Problem

Small MSME workshops in India (flour mills, lathe shops, pump stations) lose significant revenue to unexpected machine breakdowns. Existing predictive maintenance solutions assume three things that don't hold on a typical shop floor:

- **Stable internet** — Wi-Fi is unreliable or nonexistent in most workshops. Cloud-based inference is not viable.
- **Dedicated IoT sensors** — Vibration sensors and accelerometers are expensive. A small mill owner won't invest in hardware that costs more than the machine.
- **Labeled failure data** — Supervised ML models need historical examples of each failure mode. Small shops don't keep logs, let alone labeled datasets.

These constraints rule out conventional cloud-based predictive maintenance entirely.

## The Pipeline

We record the machine's operating sound using a standard microphone (phone, edge device, or USB mic). The entire pipeline runs offline on the local device. No cloud, no Wi-Fi, no internet dependency.

**Feature extraction** is done on 1-second non-overlapping windows using `librosa` and `scipy`. Five acoustic features are computed per window:

| Feature | What It Captures |
|---|---|
| RMS Energy | Vibration intensity — rises with mechanical load |
| Spectral Centroid | Center of mass of the frequency spectrum — shifts when new frequency components appear |
| Spectral Bandwidth | Spread of energy around the centroid — widens with broadband noise |
| Zero-Crossing Rate | Rate of sign changes — proxy for high-frequency content and noisiness |
| High-Frequency Energy Ratio (HFER) | Magnitude above 500 Hz / total magnitude — spikes with bearing wear, cavitation, turbulence |

The HFER feature is the most discriminating. In our MIMII validation, normal pump audio sits at HFER ~0.80, while abnormal audio drops to ~0.67, indicating a redistribution of spectral energy into the low-mid band consistent with impeller or bearing degradation.

## The Model

We use an **Isolation Forest** (`scikit-learn`, `n_estimators=100`, `contamination=0.05`) trained strictly on the normal operating baseline. The model learns the statistical envelope of healthy machine acoustics. It does not need examples of failures.

At inference, the model scores each 1-second window. When the acoustic profile drifts outside the learned normal envelope (bearing wear, hydraulic backpressure, cavitation, loose components), the anomaly score crosses the alert threshold.

**Validation (MIMII Dataset — 6 dB pump, id_00):**

| Metric | Result |
|---|---|
| True alerts (abnormal audio) | 30 / 30 windows (100%) |
| False alarms (normal audio) | 5 / 90 windows (5.6%) |

The false alarm rate is consistent with the `contamination=0.05` parameter, which models the expected ~5% noise floor in normal data.

## Dataset

[MIMII Dataset](https://zenodo.org/record/3384388) — Malfunctioning Industrial Machine Investigation and Inspection. Real-world factory recordings of pumps, fans, valves, and slide rails under normal and abnormal conditions at various SNR levels.

We use the `6_dB_pump/pump/id_00` subset:
- **Normal**: 9 clips concatenated (90 seconds total)
- **Abnormal**: 3 clips concatenated (30 seconds total)
- **Native sample rate**: 16000 Hz, mono

## How to Run

**Install dependencies:**

```bash
pip install librosa scikit-learn matplotlib soundfile scipy numpy
```

**Run the pipeline:**

```bash
python3 pipeline.py
```

**Outputs:**

- `plot1_fft_comparison.png` — FFT frequency domain comparison (normal vs. abnormal)
- `plot2_anomaly_timeline.png` — Anomaly severity score over time with alert threshold

## Project Structure

```
.
├── pipeline.py                  # Full pipeline (load → extract → train → plot)
├── mimii_normal.wav             # 90s normal pump audio (MIMII id_00)
├── mimii_abnormal.wav           # 30s abnormal pump audio (MIMII id_00)
├── plot1_fft_comparison.png     # FFT comparison output
├── plot2_anomaly_timeline.png   # Anomaly timeline output
├── 6_dB_pump/                   # Raw MIMII dataset (6 dB SNR, pump)
└── README.md
```

## Dependencies

- Python 3.8+
- numpy
- librosa
- scipy
- scikit-learn
- matplotlib
- soundfile
