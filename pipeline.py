"""
=============================================================================
 Edge-AI Acoustic Anomaly Detection Pipeline (Tech4Bharat 4.3)

 100% offline, acoustic-based early-warning system for machine breakdowns
 in small MSME workshops (flour mills, lathe shops, pump stations).

 Data source : MIMII Dataset (real-world industrial audio)
 Pipeline    : Audio → Feature Extraction → Isolation Forest → Alert
=============================================================================
"""

import os
import sys
import numpy as np
import librosa
from scipy.fft import rfft, rfftfreq

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

NORMAL_AUDIO_PATH  = 'mimii_normal.wav'
ANOMALY_AUDIO_PATH = 'mimii_abnormal.wav'

# Validate that audio files exist before doing anything
for fpath in [NORMAL_AUDIO_PATH, ANOMALY_AUDIO_PATH]:
    if not os.path.isfile(fpath):
        sys.exit(f"[ERROR] Audio file not found: {fpath}\n"
                 f"        Place your MIMII WAV files in {os.getcwd()}/")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 1 — Load Real Audio (MIMII Dataset)
#
#  • sr=None preserves the native sample rate of the recording
#  • mono=True downmixes to single channel if needed
#
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("  STEP 1 — Loading Real Audio (MIMII Dataset)")
print("=" * 72)

y_normal,  sr_normal  = librosa.load(NORMAL_AUDIO_PATH,  sr=None, mono=True)
y_anomaly, sr_anomaly = librosa.load(ANOMALY_AUDIO_PATH, sr=None, mono=True)

# Use native sample rate (should be the same for both files)
if sr_normal != sr_anomaly:
    print(f"  [WARN] Sample rate mismatch: normal={sr_normal}, anomaly={sr_anomaly}")
    print(f"         Resampling anomaly to {sr_normal} Hz...")
    y_anomaly = librosa.resample(y_anomaly, orig_sr=sr_anomaly, target_sr=sr_normal)

SR = sr_normal

dur_normal  = len(y_normal) / SR
dur_anomaly = len(y_anomaly) / SR

print(f"\n  {NORMAL_AUDIO_PATH}")
print(f"    Samples : {len(y_normal):,}")
print(f"    SR      : {SR} Hz")
print(f"    Duration: {dur_normal:.2f} s")

print(f"\n  {ANOMALY_AUDIO_PATH}")
print(f"    Samples : {len(y_anomaly):,}")
print(f"    SR      : {SR} Hz")
print(f"    Duration: {dur_anomaly:.2f} s")

print("\n── Step 1 complete. ──")


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


def extract_features(wav_path: str, sr: int = None, window_s: float = 1.0):
    """
    Load `wav_path` and extract 5 features per non-overlapping window.

    Parameters
    ----------
    wav_path  : path to a mono WAV file
    sr        : sample rate (Hz); None = use native rate
    window_s  : window length in seconds (default 1 s)

    Returns
    -------
    features : np.ndarray of shape (n_windows, 5)
               Columns: [RMS, SpectralCentroid, SpectralBW, ZCR, HFER]
    """
    y, file_sr = librosa.load(wav_path, sr=sr, mono=True)
    if sr is None:
        sr = file_sr

    win_len   = int(sr * window_s)          # samples per window
    n_windows = len(y) // win_len           # discard trailing partial window

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

normal_feats  = extract_features(NORMAL_AUDIO_PATH,  sr=SR)
anomaly_feats = extract_features(ANOMALY_AUDIO_PATH, sr=SR)

print(f"\n  {NORMAL_AUDIO_PATH}  → {normal_feats.shape[0]} windows × "
      f"{normal_feats.shape[1]} features")
print(f"  {ANOMALY_AUDIO_PATH} → {anomaly_feats.shape[0]} windows × "
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

_print_stats(NORMAL_AUDIO_PATH,  normal_feats)
_print_stats(ANOMALY_AUDIO_PATH, anomaly_feats)

print("\n── Step 2 complete. ──")


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
#  Plot 2: Anomaly severity timeline with alert threshold
#
#  Style: Clean white-background engineering-paper aesthetic.
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

# ── Clean engineering-paper plot styling ────────────────────────────────────
plt.style.use("default")
plt.rcParams.update({
    "font.size":         11,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — FFT Frequency Comparison (5-second slices)
# ════════════════════════════════════════════════════════════════════════════
SLICE_S  = 5  # seconds
slice_n  = SR * SLICE_S

# Ensure both signals are long enough for a 5-second slice
if len(y_normal) < slice_n or len(y_anomaly) < slice_n:
    # Fall back to whatever length fits
    slice_n = min(len(y_normal), len(y_anomaly))
    SLICE_S = slice_n / SR
    print(f"\n  [NOTE] Audio shorter than 5 s — using {SLICE_S:.2f} s slices.")

# Normal: first slice  |  Anomaly: last slice (deep in any fault zone)
normal_slice  = y_normal[:slice_n]
anomaly_slice = y_anomaly[-slice_n:]

# Compute magnitude spectra
freq_axis    = scipy_rfftfreq(slice_n, d=1.0 / SR)
mag_normal   = np.abs(scipy_rfft(normal_slice))
mag_anomaly  = np.abs(scipy_rfft(anomaly_slice))

# Normalize for visual comparison
mag_normal  /= (mag_normal.max() + 1e-12)
mag_anomaly /= (mag_anomaly.max() + 1e-12)

# Limit to 5000 Hz
mask_5k = freq_axis <= 5000

fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(freq_axis[mask_5k], mag_normal[mask_5k],
         color="#2e7d32", linewidth=0.7, alpha=0.85,
         label=f"Normal (first {SLICE_S:.0f} s)")
ax1.plot(freq_axis[mask_5k], mag_anomaly[mask_5k],
         color="#c62828", linewidth=0.7, alpha=0.85,
         label=f"Abnormal (last {SLICE_S:.0f} s)")

ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Normalized Magnitude")
ax1.set_title("FFT Frequency Comparison — Normal vs. Abnormal (MIMII Dataset)",
              fontsize=13, fontweight="bold")
ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax1.set_xlim(0, 5000)
ax1.set_ylim(0, 1.05)

fig1.tight_layout()
fig1.savefig("plot1_fft_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig1)
print("\n  [✓] Saved plot1_fft_comparison.png (300 dpi)")


# ════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Anomaly Severity Timeline
#
#  Concatenate normal scores + anomaly scores to simulate a continuous
#  monitoring session where a fault develops.
# ════════════════════════════════════════════════════════════════════════════
# Use however many windows are available from each file
n_normal_win  = len(normal_scores)
n_anomaly_win = len(anomaly_scores)

# For the timeline, use up to the same number of normal windows as anomaly
n_timeline = min(n_normal_win, n_anomaly_win)
timeline_normal  = normal_scores[:n_timeline]
timeline_anomaly = anomaly_scores[:n_timeline]

# Concatenate into a combined timeline
timeline_scores = np.concatenate([timeline_normal, timeline_anomaly])
time_axis = np.arange(len(timeline_scores))         # seconds

fig2, ax2 = plt.subplots(figsize=(14, 5))

# Score line
ax2.plot(time_axis, timeline_scores,
         color="#1565c0", linewidth=1.4, zorder=3, label="Anomaly Score")

# Alert threshold
ax2.axhline(THRESHOLD, color="#c62828", linewidth=1.2, linestyle="--",
            alpha=0.8, label=f"Alert Threshold ({THRESHOLD})", zorder=2)

# Shade above threshold (alert zone)
ax2.fill_between(
    time_axis, timeline_scores, THRESHOLD,
    where=(timeline_scores > THRESHOLD),
    color="#c62828", alpha=0.15, interpolate=True,
    label="Alert Zone", zorder=1,
)

# Shade below threshold (safe zone)
ax2.fill_between(
    time_axis, timeline_scores, THRESHOLD,
    where=(timeline_scores <= THRESHOLD),
    color="#2e7d32", alpha=0.08, interpolate=True, zorder=1,
)

# Mark the boundary between normal & anomaly audio
ax2.axvline(n_timeline, color="#555555", linewidth=1.0, linestyle=":",
            alpha=0.7)
y_top = timeline_scores.max() if timeline_scores.max() > 0 else 0.1
ax2.annotate("← Normal Audio | Abnormal Audio →",
             xy=(n_timeline, y_top),
             fontsize=9, color="#333333", ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999",
                       alpha=0.9))

ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Anomaly Severity Score")
ax2.set_title("Real-Time Anomaly Detection Timeline — MIMII Benchmark",
              fontsize=13, fontweight="bold")
ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)

fig2.tight_layout()
fig2.savefig("plot2_anomaly_timeline.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("  [✓] Saved plot2_anomaly_timeline.png (300 dpi)")


print("\n" + "=" * 72)
print("  Pipeline complete — all outputs saved to workspace.")
print("=" * 72)
print(f"  Audio:")
print(f"    • {NORMAL_AUDIO_PATH}")
print(f"    • {ANOMALY_AUDIO_PATH}")
print(f"  Plots:")
print(f"    • plot1_fft_comparison.png")
print(f"    • plot2_anomaly_timeline.png")
print("=" * 72)
