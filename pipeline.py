"""
=============================================================================
 STEP 1 — Synthetic Audio Generator
 Edge-AI Acoustic Anomaly Detection Pipeline (Tech4Bharat 4.3)
 
 Generates two WAV files:
   • normal_pump.wav   (90 s) – healthy motor baseline
   • anomaly_pump.wav  (30 s) – fault injection at the midpoint
=============================================================================
"""

import numpy as np
import soundfile as sf

# ── Audio parameters ────────────────────────────────────────────────────────
SR = 22050  # Sample rate (Hz) — matches librosa default

# ── Helper: generate a sine tone ────────────────────────────────────────────
def tone(freq_hz: float, duration_s: float, sr: int = SR) -> np.ndarray:
    """Pure sine wave at `freq_hz` for `duration_s` seconds."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  NORMAL PUMP  (90 seconds)
#      Healthy motor: 50 Hz fundamental + 150 Hz / 300 Hz harmonics
#      plus mild broadband white noise (σ = 0.02)
# ═══════════════════════════════════════════════════════════════════════════
NORMAL_DUR = 90  # seconds

normal  = 0.50 * tone(50,  NORMAL_DUR)   # 50 Hz mains-frequency fundamental
normal += 0.25 * tone(150, NORMAL_DUR)    # 3rd harmonic
normal += 0.10 * tone(300, NORMAL_DUR)    # 6th harmonic

# Mild mechanical noise floor
rng = np.random.default_rng(seed=42)
normal += 0.02 * rng.standard_normal(len(normal))

# Peak-normalize to ±0.9 to leave headroom
normal = 0.9 * (normal / np.max(np.abs(normal)))

sf.write("normal_pump.wav", normal, SR, subtype="PCM_16")
print(f"[✓] normal_pump.wav  — {NORMAL_DUR}s, {len(normal)} samples, SR={SR}")


# ═══════════════════════════════════════════════════════════════════════════
#  2.  ANOMALY PUMP  (30 seconds)
#      First half  (0–15 s)  : identical healthy motor profile
#      Second half (15–30 s) : fault injection —
#          • High-frequency turbulence at 850 Hz and 1800 Hz
#          • +40 % load amplitude ramp (simulates hydraulic backpressure)
#          • Increased noise floor (σ = 0.06)
# ═══════════════════════════════════════════════════════════════════════════
ANOM_DUR = 30
HALF     = ANOM_DUR // 2   # fault onset at 15 s

# --- healthy segment (first half) ---
seg_ok  = 0.50 * tone(50,  HALF)
seg_ok += 0.25 * tone(150, HALF)
seg_ok += 0.10 * tone(300, HALF)
seg_ok += 0.02 * rng.standard_normal(len(seg_ok))

# --- faulty segment (second half) ---
n_fault = int(SR * HALF)
t_fault = np.linspace(0, HALF, n_fault, endpoint=False)

# Base motor (same harmonics)
seg_fault  = 0.50 * np.sin(2 * np.pi * 50  * t_fault)
seg_fault += 0.25 * np.sin(2 * np.pi * 150 * t_fault)
seg_fault += 0.10 * np.sin(2 * np.pi * 300 * t_fault)

# ⚠ Fault signatures
seg_fault += 0.30 * np.sin(2 * np.pi * 850  * t_fault)   # bearing-wear tone
seg_fault += 0.20 * np.sin(2 * np.pi * 1800 * t_fault)   # high-freq turbulence

# ⚠ Load amplitude ramp: linearly grows from 1.0× to 1.4× over the segment
load_ramp = np.linspace(1.0, 1.4, n_fault)
seg_fault *= load_ramp

# ⚠ Elevated noise floor
seg_fault += 0.06 * rng.standard_normal(n_fault)

# --- stitch & normalize ---
anomaly = np.concatenate([seg_ok, seg_fault])
anomaly = 0.9 * (anomaly / np.max(np.abs(anomaly)))

sf.write("anomaly_pump.wav", anomaly, SR, subtype="PCM_16")
print(f"[✓] anomaly_pump.wav — {ANOM_DUR}s, {len(anomaly)} samples, SR={SR}")

print("\n── Step 1 complete. Ready for Step 2 (Feature Extraction). ──")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 2 — Feature Extraction (1-second rolling windows)
#
#  For each 1-second window we compute 5 mechanical acoustic features:
#    1. RMS Energy           – vibration intensity proxy
#    2. Spectral Centroid    – centre of mass of the frequency spectrum
#    3. Spectral Bandwidth   – spread of energy around the centroid
#    4. Zero-Crossing Rate   – proxy for noisiness / high-freq content
#    5. High-Freq Energy Ratio (HFER) – |FFT(>500 Hz)| / |FFT(total)|
#
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

import librosa
from scipy.fft import rfft, rfftfreq


def extract_features(wav_path: str, sr: int = SR, window_s: float = 1.0):
    """
    Load `wav_path` and extract 5 features per non-overlapping window.

    Parameters
    ----------
    wav_path  : path to a mono WAV file
    sr        : sample rate (Hz)
    window_s  : window length in seconds (default 1 s)

    Returns
    -------
    features : np.ndarray of shape (n_windows, 5)
               Columns: [RMS, SpectralCentroid, SpectralBW, ZCR, HFER]
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    win_len  = int(sr * window_s)          # samples per window
    n_windows = len(y) // win_len          # discard trailing partial window

    feature_rows = []

    for i in range(n_windows):
        frame = y[i * win_len : (i + 1) * win_len]

        # ── 1. RMS Energy ───────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(frame ** 2)))

        # ── 2. Spectral Centroid (Hz) ───────────────────────────────────
        sc = librosa.feature.spectral_centroid(
            y=frame, sr=sr, n_fft=win_len, hop_length=win_len
        )
        spectral_centroid = float(sc.mean())

        # ── 3. Spectral Bandwidth (Hz) ─────────────────────────────────
        sb = librosa.feature.spectral_bandwidth(
            y=frame, sr=sr, n_fft=win_len, hop_length=win_len
        )
        spectral_bw = float(sb.mean())

        # ── 4. Zero-Crossing Rate ──────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(
            y=frame, frame_length=win_len, hop_length=win_len
        )
        zero_crossing = float(zcr.mean())

        # ── 5. High-Frequency Energy Ratio (HFER) ─────────────────────
        #    Ratio of spectral magnitude above 500 Hz to total magnitude.
        #    Rises sharply when bearing-wear / turbulence tones appear.
        fft_mag  = np.abs(rfft(frame))
        fft_freq = rfftfreq(len(frame), d=1.0 / sr)
        hf_mask  = fft_freq > 500.0
        hfer = float(fft_mag[hf_mask].sum() / (fft_mag.sum() + 1e-12))

        feature_rows.append([rms, spectral_centroid, spectral_bw,
                             zero_crossing, hfer])

    features = np.array(feature_rows, dtype=np.float64)
    return features


# ── Run extraction on both WAV files ────────────────────────────────────────
FEATURE_NAMES = ["RMS", "SpectralCentroid", "SpectralBW", "ZCR", "HFER"]

print("\n" + "=" * 72)
print("  STEP 2 — Feature Extraction (1-second windows)")
print("=" * 72)

normal_feats  = extract_features("normal_pump.wav")
anomaly_feats = extract_features("anomaly_pump.wav")

print(f"\n  normal_pump.wav  → {normal_feats.shape[0]} windows × "
      f"{normal_feats.shape[1]} features")
print(f"  anomaly_pump.wav → {anomaly_feats.shape[0]} windows × "
      f"{anomaly_feats.shape[1]} features")

# ── Print summary statistics ────────────────────────────────────────────────
def _print_stats(name: str, feats: np.ndarray):
    print(f"\n  ┌─ {name} ─{'─' * (55 - len(name))}┐")
    print(f"  │ {'Feature':<20s} {'Mean':>10s} {'Std':>10s} "
          f"{'Min':>10s} {'Max':>10s} │")
    print(f"  │{'─' * 62}│")
    for j, fn in enumerate(FEATURE_NAMES):
        col = feats[:, j]
        print(f"  │ {fn:<20s} {col.mean():10.5f} {col.std():10.5f} "
              f"{col.min():10.5f} {col.max():10.5f} │")
    print(f"  └{'─' * 62}┘")

_print_stats("normal_pump.wav",  normal_feats)
_print_stats("anomaly_pump.wav", anomaly_feats)

print("\n── Step 2 complete. Ready for Step 3 (Model Training). ──")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 3 — Isolation Forest Training & Anomaly Scoring
#
#  • StandardScaler fitted on normal features only (zero-mean, unit-var)
#  • IsolationForest trained STRICTLY on normal operating data
#  • Scores inverted so higher value = higher anomaly severity
#
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

print("\n" + "=" * 72)
print("  STEP 3 — Isolation Forest Training & Scoring")
print("=" * 72)

# ── 3a. Scale features (fit on normal only) ─────────────────────────────────
scaler = StandardScaler()
normal_scaled  = scaler.fit_transform(normal_feats)   # fit + transform
anomaly_scaled = scaler.transform(anomaly_feats)       # transform only

print(f"\n  Scaler fitted on {normal_scaled.shape[0]} normal windows.")
print(f"  Normal  scaled shape : {normal_scaled.shape}")
print(f"  Anomaly scaled shape : {anomaly_scaled.shape}")

# ── 3b. Train Isolation Forest on normal data ──────────────────────────────
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
)
iso_forest.fit(normal_scaled)
print(f"\n  IsolationForest trained (n_estimators=100, contamination=0.05).")

# ── 3c. Anomaly scoring ────────────────────────────────────────────────────
#   decision_function: positive = inlier, negative = outlier
#   We invert (×-1) so higher = more anomalous (severity score)
normal_scores  = -1 * iso_forest.decision_function(normal_scaled)
anomaly_scores = -1 * iso_forest.decision_function(anomaly_scaled)

print(f"\n  Normal  scores — mean: {normal_scores.mean():+.4f},  "
      f"max: {normal_scores.max():+.4f}")
print(f"  Anomaly scores — mean: {anomaly_scores.mean():+.4f},  "
      f"max: {anomaly_scores.max():+.4f}")

# Quick classification summary
THRESHOLD = 0.0
n_normal_alerts  = int((normal_scores > THRESHOLD).sum())
n_anomaly_alerts = int((anomaly_scores > THRESHOLD).sum())
print(f"\n  Alert threshold : {THRESHOLD}")
print(f"  False alarms (normal data)  : {n_normal_alerts} / {len(normal_scores)} "
      f"windows")
print(f"  True  alerts (anomaly data) : {n_anomaly_alerts} / {len(anomaly_scores)} "
      f"windows")

print("\n── Step 3 complete. ──")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 4 — Visualization & Output
#
#  Plot 1: FFT frequency comparison (normal vs. anomaly)
#  Plot 2: 60-second anomaly severity timeline with alert threshold
#
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
from scipy.fft import rfft as scipy_rfft, rfftfreq as scipy_rfftfreq

print("\n" + "=" * 72)
print("  STEP 4 — Visualization & Output")
print("=" * 72)

# ── Shared plot styling ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f0f0f",
    "axes.facecolor":    "#0f0f0f",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#cccccc",
    "text.color":        "#cccccc",
    "xtick.color":       "#999999",
    "ytick.color":       "#999999",
    "grid.color":        "#222222",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#1a1a1a",
    "legend.edgecolor":  "#333333",
    "font.size":         11,
})

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — FFT Frequency Comparison (5-second slices)
# ════════════════════════════════════════════════════════════════════════════
# Load raw audio for FFT
y_normal, _  = librosa.load("normal_pump.wav",  sr=SR, mono=True)
y_anomaly, _ = librosa.load("anomaly_pump.wav", sr=SR, mono=True)

SLICE_S  = 5  # seconds
slice_n  = SR * SLICE_S

# Normal: first 5 seconds  |  Anomaly: last 5 seconds (deep in the fault zone)
normal_slice  = y_normal[:slice_n]
anomaly_slice = y_anomaly[-slice_n:]

# Compute magnitude spectra
freq_axis    = scipy_rfftfreq(slice_n, d=1.0 / SR)
mag_normal   = np.abs(scipy_rfft(normal_slice))
mag_anomaly  = np.abs(scipy_rfft(anomaly_slice))

# Normalize for visual comparison
mag_normal  /= mag_normal.max()
mag_anomaly /= mag_anomaly.max()

# Limit to 5000 Hz
mask_5k = freq_axis <= 5000

fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(freq_axis[mask_5k], mag_normal[mask_5k],
         color="#00e676", linewidth=0.8, alpha=0.85, label="Normal (0–5 s)")
ax1.plot(freq_axis[mask_5k], mag_anomaly[mask_5k],
         color="#ff1744", linewidth=0.8, alpha=0.85, label="Anomaly (25–30 s)")

# Annotate known fault frequencies
for f_hz, lbl in [(850, "850 Hz\nbearing wear"), (1800, "1800 Hz\nturbulence")]:
    ax1.axvline(f_hz, color="#ffd740", linewidth=0.9, linestyle=":", alpha=0.7)
    ax1.annotate(lbl, xy=(f_hz, 0.92), fontsize=8, color="#ffd740",
                 ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a1a", ec="#ffd740",
                           alpha=0.8))

ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Normalized Magnitude")
ax1.set_title("FFT Frequency Comparison — Normal vs. Anomaly",
              fontsize=14, fontweight="bold", color="white")
ax1.legend(loc="upper right", fontsize=10)
ax1.set_xlim(0, 5000)
ax1.set_ylim(0, 1.05)
ax1.grid(True)

fig1.tight_layout()
fig1.savefig("plot1_fft_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig1)
print("\n  [✓] Saved plot1_fft_comparison.png (300 dpi)")


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Anomaly Severity Timeline (60 seconds)
#
#  Layout: first 30 s = normal machine → next 30 s = anomaly machine
#  This simulates a continuous monitoring session where a fault develops.
# ════════════════════════════════════════════════════════════════════════════
# Take first 30 windows from normal scores
n_timeline = 30
timeline_normal  = normal_scores[:n_timeline]
timeline_anomaly = anomaly_scores                   # already 30 windows

# Concatenate into a 60-second timeline
timeline_scores = np.concatenate([timeline_normal, timeline_anomaly])
time_axis = np.arange(len(timeline_scores))         # seconds

fig2, ax2 = plt.subplots(figsize=(14, 5))

# Score line
ax2.plot(time_axis, timeline_scores,
         color="#42a5f5", linewidth=1.6, zorder=3, label="Anomaly Score")

# Alert threshold
ax2.axhline(THRESHOLD, color="#ff1744", linewidth=1.2, linestyle="--",
            alpha=0.9, label=f"Alert Threshold ({THRESHOLD})", zorder=2)

# Shade red above threshold
ax2.fill_between(
    time_axis, timeline_scores, THRESHOLD,
    where=(timeline_scores > THRESHOLD),
    color="#ff1744", alpha=0.25, interpolate=True,
    label="Alert Zone", zorder=1,
)

# Shade green below threshold
ax2.fill_between(
    time_axis, timeline_scores, THRESHOLD,
    where=(timeline_scores <= THRESHOLD),
    color="#00e676", alpha=0.10, interpolate=True, zorder=1,
)

# Mark the boundary between normal & anomaly audio
ax2.axvline(n_timeline, color="#ffd740", linewidth=1.0, linestyle=":",
            alpha=0.8)
ax2.annotate("← Normal Machine | Anomaly Machine →",
             xy=(n_timeline, ax2.get_ylim()[1] if timeline_scores.max() > 0
                 else 0.1),
             fontsize=9, color="#ffd740", ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec="#ffd740",
                       alpha=0.8))

ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Anomaly Severity Score")
ax2.set_title("Real-Time Anomaly Detection Timeline — Early Warning System",
              fontsize=14, fontweight="bold", color="white")
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True)

fig2.tight_layout()
fig2.savefig("plot2_anomaly_timeline.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("  [✓] Saved plot2_anomaly_timeline.png (300 dpi)")


print("\n" + "=" * 72)
print("  Pipeline complete — all outputs saved to workspace.")
print("=" * 72)
print("  Files:")
print("    • normal_pump.wav")
print("    • anomaly_pump.wav")
print("    • plot1_fft_comparison.png")
print("    • plot2_anomaly_timeline.png")
print("=" * 72)
