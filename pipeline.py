"""
=============================================================================
 Edge-AI Acoustic Anomaly Detection Pipeline (Tech4Bharat 4.3)

 100% offline, acoustic-based early-warning system for machine breakdowns
 in small MSME workshops (flour mills, lathe shops, pump stations).

 Data source : MIMII Dataset (real-world industrial audio, 6 dB pump id_00)
 Pipeline    : Audio → Feature Extraction → Isolation Forest → Alert
 Validation  : Proper train/test split on individual 10-second clips.
               Preliminary validation on a single pump type.
=============================================================================
"""

import os
import sys
import glob
import numpy as np
import librosa
from datetime import datetime, timedelta
from scipy.fft import rfft, rfftfreq

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MIMII_PUMP_DIR = '6_dB_pump/pump/id_00'
NORMAL_DIR     = os.path.join(MIMII_PUMP_DIR, 'normal')
ABNORMAL_DIR   = os.path.join(MIMII_PUMP_DIR, 'abnormal')

# Train/test split parameters
N_TRAIN_NORMAL  = 60     # clips used for training (60 clips × 10 s = 600 s)
N_TEST_NORMAL   = 30     # held-out normal clips for testing
N_TEST_ABNORMAL = 30     # abnormal clips for testing

# Validate dataset exists
for d in [NORMAL_DIR, ABNORMAL_DIR]:
    if not os.path.isdir(d):
        sys.exit(f"[ERROR] Directory not found: {d}\n"
                 f"        Place the MIMII 6_dB_pump dataset in {os.getcwd()}/")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 1 — Load & Split MIMII Clips (Train/Test)
#
#  Training set  : N_TRAIN_NORMAL normal clips (model learns baseline)
#  Test set      : N_TEST_NORMAL normal + N_TEST_ABNORMAL abnormal clips
#                  (all unseen by the model)
#
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("  STEP 1 — Load & Split MIMII Clips (Train/Test)")
print("=" * 72)

normal_files   = sorted(glob.glob(os.path.join(NORMAL_DIR,   '*.wav')))
abnormal_files = sorted(glob.glob(os.path.join(ABNORMAL_DIR, '*.wav')))

print(f"\n  Dataset: {MIMII_PUMP_DIR}")
print(f"    Total normal clips   : {len(normal_files)}")
print(f"    Total abnormal clips : {len(abnormal_files)}")

# Shuffle with fixed seed for reproducibility, then split
rng = np.random.default_rng(seed=42)
rng.shuffle(normal_files)
rng.shuffle(abnormal_files)

train_normal_files = normal_files[:N_TRAIN_NORMAL]
test_normal_files  = normal_files[N_TRAIN_NORMAL : N_TRAIN_NORMAL + N_TEST_NORMAL]
test_abnormal_files = abnormal_files[:N_TEST_ABNORMAL]

print(f"\n  Train set (normal only) : {len(train_normal_files)} clips")
print(f"  Test set  (normal)      : {len(test_normal_files)} clips")
print(f"  Test set  (abnormal)    : {len(test_abnormal_files)} clips")

# Detect native sample rate from the first file
_, SR = librosa.load(train_normal_files[0], sr=None, mono=True, duration=0.1)
SR = int(SR)
print(f"  Native sample rate      : {SR} Hz")

print("\n── Step 1 complete. ──")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 2 — Feature Extraction (1-second rolling windows)
#
#  For each 1-second window we compute 5 mechanical acoustic features:
#    1. RMS Energy            – vibration intensity proxy
#    2. Spectral Centroid     – centre of mass of the frequency spectrum
#    3. Spectral Bandwidth    – spread of energy around the centroid
#    4. Zero-Crossing Rate    – proxy for noisiness / high-freq content
#    5. High-Freq Energy Ratio (HFER) – raw ratio of spectral magnitude
#                               above 500 Hz to total magnitude
#
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════


def extract_features_from_signal(y, sr: int, window_s: float = 1.0):
    """
    Extract 5 features per non-overlapping window from a raw audio signal.

    Returns
    -------
    features : np.ndarray of shape (n_windows, 5)
               Columns: [RMS, SpectralCentroid, SpectralBW, ZCR, HFER]
    """
    win_len   = int(sr * window_s)
    n_windows = len(y) // win_len

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
        #    Raw ratio of spectral magnitude above 500 Hz to total.
        fft_mag  = np.abs(rfft(frame))
        fft_freq = rfftfreq(len(frame), d=1.0 / sr)
        hf_mask  = fft_freq > 500.0
        hfer = float(fft_mag[hf_mask].sum() / (fft_mag.sum() + 1e-12))

        feature_rows.append([rms, spectral_centroid, spectral_bw,
                             zero_crossing, hfer])

    return np.array(feature_rows, dtype=np.float64)


def extract_features_from_clips(file_list, sr: int, label: str = ""):
    """
    Extract features from a list of WAV clip paths.

    Returns
    -------
    all_features : np.ndarray of shape (total_windows, 5)
    clip_boundaries : list of (start_idx, end_idx) per clip
    """
    all_rows = []
    clip_boundaries = []
    for i, fpath in enumerate(file_list):
        y, _ = librosa.load(fpath, sr=sr, mono=True)
        feats = extract_features_from_signal(y, sr)
        start = len(all_rows)
        all_rows.append(feats)
        clip_boundaries.append((start, start + len(feats)))

    all_features = np.vstack(all_rows) if all_rows else np.empty((0, 5))
    if label:
        print(f"    {label}: {len(file_list)} clips → "
              f"{all_features.shape[0]} windows × {all_features.shape[1]} features")
    return all_features, clip_boundaries


FEATURE_NAMES = ["RMS", "SpectralCentroid", "SpectralBW", "ZCR", "HFER"]

print("\n" + "=" * 72)
print("  STEP 2 — Feature Extraction (1-second windows)")
print("=" * 72)

print("\n  Extracting training features...")
train_feats, train_bounds = extract_features_from_clips(
    train_normal_files, SR, label="Train (normal)")

print("  Extracting test features...")
test_normal_feats, test_normal_bounds = extract_features_from_clips(
    test_normal_files, SR, label="Test  (normal)")
test_abnormal_feats, test_abnormal_bounds = extract_features_from_clips(
    test_abnormal_files, SR, label="Test  (abnormal)")

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

_print_stats("Train (normal)", train_feats)
_print_stats("Test (normal)",  test_normal_feats)
_print_stats("Test (abnormal)", test_abnormal_feats)

print("\n── Step 2 complete. ──")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 3 — Isolation Forest Training & Clip-Level Validation
#
#  • StandardScaler fitted on training normal features only
#  • IsolationForest trained STRICTLY on training normal data
#  • Evaluated on held-out test clips (both normal and abnormal)
#  • Clip-level classification: a clip is flagged as anomalous if
#    the majority (>50%) of its 1-second windows exceed the threshold
#
# Preliminary validation on a single pump type.
# Dataset: id_00 pump, 6 dB SNR.
#
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

print("\n" + "=" * 72)
print("  STEP 3 — Isolation Forest Training & Clip-Level Validation")
print("=" * 72)

# ── 3a. Scale features (fit on training normal only) ────────────────────────
scaler = StandardScaler()
train_scaled        = scaler.fit_transform(train_feats)
test_normal_scaled  = scaler.transform(test_normal_feats)
test_abnormal_scaled = scaler.transform(test_abnormal_feats)

print(f"\n  Scaler fitted on {train_scaled.shape[0]} training windows "
      f"({len(train_normal_files)} clips).")

# ── 3b. Train Isolation Forest on training normal data ─────────────────────
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
)
iso_forest.fit(train_scaled)
print(f"  IsolationForest trained (n_estimators=100, contamination=0.05).")

# ── 3c. Score test data ────────────────────────────────────────────────────
#   decision_function: positive = inlier, negative = outlier
#   We invert (×-1) so higher = more anomalous (severity score)
THRESHOLD = 0.0

test_normal_scores  = -1 * iso_forest.decision_function(test_normal_scaled)
test_abnormal_scores = -1 * iso_forest.decision_function(test_abnormal_scaled)

# Also score training data for timeline visualization
train_scores = -1 * iso_forest.decision_function(train_scaled)

# ── 3d. Clip-level classification ──────────────────────────────────────────
#   A clip is classified as anomalous if >50% of its windows are above threshold.
def classify_clips(scores, boundaries):
    """Return number of clips classified as anomalous."""
    n_flagged = 0
    for start, end in boundaries:
        clip_scores = scores[start:end]
        if (clip_scores > THRESHOLD).sum() > len(clip_scores) / 2:
            n_flagged += 1
    return n_flagged

normal_clips_flagged  = classify_clips(test_normal_scores, test_normal_bounds)
abnormal_clips_flagged = classify_clips(test_abnormal_scores, test_abnormal_bounds)

n_test_normal  = len(test_normal_bounds)
n_test_abnormal = len(test_abnormal_bounds)

clip_accuracy_normal  = (n_test_normal - normal_clips_flagged) / n_test_normal
clip_accuracy_abnormal = abnormal_clips_flagged / n_test_abnormal
overall_accuracy = ((n_test_normal - normal_clips_flagged) + abnormal_clips_flagged) \
                   / (n_test_normal + n_test_abnormal)

print(f"\n  ── Clip-Level Validation Results ──")
print(f"  (Preliminary validation on a single pump type: id_00, 6 dB SNR)")
print(f"\n  Test normal clips correctly passed : "
      f"{n_test_normal - normal_clips_flagged} / {n_test_normal} "
      f"({clip_accuracy_normal:.1%})")
print(f"  Test abnormal clips correctly flagged : "
      f"{abnormal_clips_flagged} / {n_test_abnormal} "
      f"({clip_accuracy_abnormal:.1%})")
print(f"\n  Overall clip-level accuracy : {overall_accuracy:.1%}")
print(f"  False alarm rate (clip)     : "
      f"{normal_clips_flagged}/{n_test_normal} "
      f"({normal_clips_flagged/n_test_normal:.1%})")
print(f"  Detection rate (clip)       : "
      f"{abnormal_clips_flagged}/{n_test_abnormal} "
      f"({abnormal_clips_flagged/n_test_abnormal:.1%})")

# ── Window-level summary for reference ─────────────────────────────────────
n_win_normal_alerts  = int((test_normal_scores > THRESHOLD).sum())
n_win_abnormal_alerts = int((test_abnormal_scores > THRESHOLD).sum())
print(f"\n  ── Window-Level Summary ──")
print(f"  Normal  windows above threshold : {n_win_normal_alerts} / "
      f"{len(test_normal_scores)}")
print(f"  Abnormal windows above threshold: {n_win_abnormal_alerts} / "
      f"{len(test_abnormal_scores)}")

print("\n── Step 3 complete. ──")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 4 — Visualization & Output
#
#  Plot 1: FFT frequency comparison (normal vs. abnormal)
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
# Load one normal and one abnormal clip for FFT visualization
y_normal_viz, _  = librosa.load(test_normal_files[0],  sr=SR, mono=True)
y_abnormal_viz, _ = librosa.load(test_abnormal_files[0], sr=SR, mono=True)

SLICE_S  = 5  # seconds
slice_n  = SR * SLICE_S

if len(y_normal_viz) < slice_n or len(y_abnormal_viz) < slice_n:
    slice_n = min(len(y_normal_viz), len(y_abnormal_viz))
    SLICE_S = slice_n / SR

normal_slice  = y_normal_viz[:slice_n]
abnormal_slice = y_abnormal_viz[:slice_n]

# Compute magnitude spectra
freq_axis    = scipy_rfftfreq(slice_n, d=1.0 / SR)
mag_normal   = np.abs(scipy_rfft(normal_slice))
mag_abnormal = np.abs(scipy_rfft(abnormal_slice))

# Normalize for visual comparison
mag_normal   /= (mag_normal.max() + 1e-12)
mag_abnormal /= (mag_abnormal.max() + 1e-12)

# Limit to 5000 Hz
mask_5k = freq_axis <= 5000

fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(freq_axis[mask_5k], mag_normal[mask_5k],
         color="#2e7d32", linewidth=0.7, alpha=0.85,
         label=f"Normal (first {SLICE_S:.0f} s)")
ax1.plot(freq_axis[mask_5k], mag_abnormal[mask_5k],
         color="#c62828", linewidth=0.7, alpha=0.85,
         label=f"Abnormal (first {SLICE_S:.0f} s)")

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
#  Concatenate test normal scores + test abnormal scores to simulate a
#  continuous monitoring session where a fault develops.
# ════════════════════════════════════════════════════════════════════════════
n_timeline = min(len(test_normal_scores), len(test_abnormal_scores))
timeline_normal  = test_normal_scores[:n_timeline]
timeline_abnormal = test_abnormal_scores[:n_timeline]

timeline_scores = np.concatenate([timeline_normal, timeline_abnormal])
time_axis = np.arange(len(timeline_scores))

fig2, ax2 = plt.subplots(figsize=(14, 5))

ax2.plot(time_axis, timeline_scores,
         color="#1565c0", linewidth=1.4, zorder=3, label="Anomaly Score")

ax2.axhline(THRESHOLD, color="#c62828", linewidth=1.2, linestyle="--",
            alpha=0.8, label=f"Alert Threshold ({THRESHOLD})", zorder=2)

ax2.fill_between(
    time_axis, timeline_scores, THRESHOLD,
    where=(timeline_scores > THRESHOLD),
    color="#c62828", alpha=0.15, interpolate=True,
    label="Alert Zone", zorder=1,
)

ax2.fill_between(
    time_axis, timeline_scores, THRESHOLD,
    where=(timeline_scores <= THRESHOLD),
    color="#2e7d32", alpha=0.08, interpolate=True, zorder=1,
)

ax2.axvline(n_timeline, color="#555555", linewidth=1.0, linestyle=":",
            alpha=0.7)
y_top = timeline_scores.max() if timeline_scores.max() > 0 else 0.1
ax2.annotate("← Normal (test) | Abnormal (test) →",
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


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
#  STEP 5 — Agentic AI: Maintenance Scheduler
#
#  If an anomaly is detected during a monitoring session, this function
#  calculates a 30-minute idle window after the shift ends and drafts a
#  WhatsApp-style maintenance alert for the shop mechanic.
#
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════


def agentic_maintenance_scheduler(anomaly_detected: bool,
                                  current_time: datetime,
                                  shift_end_time: datetime) -> str:
    """
    Agentic maintenance scheduling logic.

    If an anomaly is detected, schedules a 30-minute maintenance window
    beginning at shift_end_time and returns a formatted WhatsApp-style
    message for the mechanic.

    Parameters
    ----------
    anomaly_detected : bool
        Whether the anomaly detection model flagged a fault.
    current_time : datetime
        Timestamp when the anomaly was detected.
    shift_end_time : datetime
        When the current shift ends.

    Returns
    -------
    message : str
        Formatted maintenance alert (WhatsApp draft), or a no-action note.
    """
    if not anomaly_detected:
        return (f"[{current_time.strftime('%H:%M')}] "
                f"All readings normal. No maintenance action required.")

    maintenance_end = shift_end_time + timedelta(minutes=30)
    maintenance_window = (f"{shift_end_time.strftime('%H:%M')} – "
                          f"{maintenance_end.strftime('%H:%M')}")

    message = (
        f"--- MAINTENANCE ALERT ---\n"
        f"\n"
        f"Machine: Pump Unit (Shop Floor)\n"
        f"Status: Acoustic anomaly detected\n"
        f"Detected at: {current_time.strftime('%d-%b-%Y %H:%M')}\n"
        f"\n"
        f"The vibration profile has shifted outside the normal operating\n"
        f"envelope. Possible causes: bearing wear, impeller imbalance,\n"
        f"or hydraulic backpressure.\n"
        f"\n"
        f"Suggested 30-minute maintenance window: {maintenance_window}\n"
        f"\n"
        f"Action: Inspect bearings, check alignment, listen for\n"
        f"abnormal rattle or grinding before next shift.\n"
        f"\n"
        f"--- Sent via Edge-AI Monitor ---"
    )
    return message


# ── Demo the agentic scheduler ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("  STEP 5 — Agentic AI: Maintenance Scheduler")
print("=" * 72)

# Simulate: anomaly detected now, shift ends at 18:00
demo_current_time  = datetime.now()
demo_shift_end     = demo_current_time.replace(hour=18, minute=0, second=0)

# If current time is past 18:00, set shift end to tomorrow
if demo_current_time > demo_shift_end:
    demo_shift_end += timedelta(days=1)

# Was an anomaly detected in our test run?
pipeline_anomaly_detected = abnormal_clips_flagged > 0

msg = agentic_maintenance_scheduler(
    anomaly_detected=pipeline_anomaly_detected,
    current_time=demo_current_time,
    shift_end_time=demo_shift_end,
)
print(f"\n  Anomaly detected: {pipeline_anomaly_detected}")
print(f"\n  Drafted WhatsApp message:\n")
for line in msg.split('\n'):
    print(f"    {line}")


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  Pipeline complete — all outputs saved to workspace.")
print("=" * 72)
print(f"  Plots:")
print(f"    • plot1_fft_comparison.png")
print(f"    • plot2_anomaly_timeline.png")
print(f"  Validation:")
print(f"    • Overall clip-level accuracy: {overall_accuracy:.1%}")
print(f"    • Detection rate: {clip_accuracy_abnormal:.1%}")
print(f"    • False alarm rate: {normal_clips_flagged/n_test_normal:.1%}")
print("=" * 72)
