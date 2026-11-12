# ToneGauge - Edge-Acoustic Scanner for MSMEs

## Problem

Small workshops run machines to failure because existing IoT solutions require Wi-Fi, expensive sensors, and labeled failure data.

## Solution

ToneGauge is an offline-first app using the phone's microphone. The system starts with a short baseline and refines it over days.

## Pipeline (Built)

The pipeline extracts acoustic features offline, including RMS energy and spectral centroids. An Isolation Forest is trained purely on normal baseline audio. The current evaluation uses a separate holdout set from the MIMII pump dataset: 60 normal clips for training, 30 normal clips for testing, and 30 abnormal clips for testing. Results are reported as clip-level accuracy.

A Python-based scheduling agent drafts maintenance alerts based on shift times. When an anomaly is detected, it proposes a 30-minute maintenance window starting at the shift end time.

## UI (Designed Prototype)

Illustrative wireframes demonstrate the operator workflow.

## Limitations & Next Steps

Current tests are limited to a single MIMII pump type. Device microphone differences, ambient noise variations, and load states require separate profiles. The next phase is field validation at local mills in Coimbatore.

## How to run

The pipeline expects the MIMII clips under `6_dB_pump/pump/id_00/normal` and `6_dB_pump/pump/id_00/abnormal`.

```bash
python3 -m pip install numpy librosa scipy scikit-learn matplotlib soundfile
python3 pipeline.py
```
