# ToneGauge: Offline Acoustic Early-Warning for MSMEs
*Tech4Bharat 4.3 (Agentic AI)*

![FFT comparison](plot1_fft_comparison.png)
![Anomaly timeline](plot2_anomaly_timeline.png)

*Example clips from the test set, not the full evaluation.*

## Problem
Small workshops run machines until they fail. Existing predictive-maintenance tools need cloud connectivity, dedicated sensors and labelled failure data, which small shops don't have.

## Solution
ToneGauge listens to a machine with a phone microphone, learns what normal sounds like, and flags drift. It drafts a maintenance plan for the owner, who approves every action.

**Status:** the prototype runs as a Python pipeline. It is designed for on-device use. Android packaging and baseline refinement over days are planned, not built.

## Pipeline (built)
- 1-second non-overlapping windows, 16 kHz mono
- Five features per window: RMS energy, spectral centroid, spectral bandwidth, zero-crossing rate, high-frequency energy ratio (HFER). Standardised with a scaler fitted on training windows only.
- Isolation Forest (`n_estimators=100`, `contamination=0.05`, `random_state=42`), trained only on normal clips
- Clip rule: a clip is flagged abnormal if more than 50% of its 1-second windows are above the anomaly threshold. The threshold comes from the model fitted on training data only.

## Evaluation (MIMII, 6 dB pump, id_00)
MIMII mixes real factory noise into the machine sound (6 dB SNR). Pump id_00 has 1,006 normal and 143 abnormal clips. We split randomly by clip (seed 42), so no clip appears in both train and test.

| Set | Clips | Result |
|---|---|---|
| Train (normal) | 60 | n/a |
| Test, normal | 30 | 0/30 falsely flagged |
| Test, abnormal | 30 | 19/30 detected (63%) |

Window level: 24/300 normal windows (8%) and 232/300 abnormal windows (77%) scored above the threshold. On average, abnormal test clips have about twice the RMS energy and an ~8% lower spectral centroid than normal ones.

Preliminary: one machine ID and 30 clips per class, so the rates are rough. The clip rule is conservative: no false alarms, but 11 of 30 abnormal clips were missed.

## Scheduling agent (built)
Given a detected anomaly and the shift end time entered by the owner, it proposes a 60-minute maintenance window starting 30 minutes after shift end and drafts a message to the mechanic. The message is a draft: nothing is sent or stopped without owner approval.

## AI components
- **Learning model:** an Isolation Forest (unsupervised machine learning) learns what normal sounds like from healthy audio only. No failure labels are needed.
- **Agent:** a rule-based scheduler turns an anomaly and the owner's shift end into a suggested maintenance window and a drafted message. It runs the loop scan, flag, propose, wait for approval. It does not use a large language model, so it works offline.

## UI (designed prototype)
Illustrative wireframes with placeholder values. Not connected to the pipeline.

## Limitations & next steps
- One pump ID only (id_00), 30 test clips per class.
- Detection is partial (19/30). Mild faults are likely missed.
- Phone microphone differences, background noise and load states need separate profiles.
- Not yet tested on real shop-floor recordings.
- Next: more MIMII IDs and machine types, then field validation at local mills in Coimbatore.

## How to run
Place MIMII clips under `6_dB_pump/pump/id_00/normal` and `.../abnormal` (not included; download from the dataset link).
```bash
python3 -m pip install numpy librosa scipy scikit-learn matplotlib soundfile
python3 pipeline.py
```
`mimii_normal.wav` and `mimii_abnormal.wav` are short demo clips. The evaluation uses the raw clips.

## Dataset
MIMII Dataset (Hitachi), Zenodo record 3384388, CC BY-SA 4.0. Purohit et al., "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection", DCASE 2019.
