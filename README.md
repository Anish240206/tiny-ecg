# tinyecg

Reconstruction-based ECG anomaly detection for an ultra-low-power edge node.
A tiny 1D convolutional autoencoder is trained on **normal heartbeats only** (MIT-BIH), quantized to
**INT8 (14.4 KB)**, and run in a **continuous sliding-window loop** (encode, decode, score, alarm) designed
for an ESP32-S3 + AD8232 wearable with TensorFlow Lite for Microcontrollers.

Course project, BCSE306L (AI), VIT Vellore.

## Results (MIT-BIH, inter-patient: train DS1, test DS2, 39,590 test windows, 35.5% anomalous)

| Model variant | AUROC |
|---|---|
| Dense-bottleneck autoencoder (offline filter) | 0.47 |
| Conv autoencoder, 2-channel bottleneck (offline filter) | 0.75 |
| Conv autoencoder, 1-channel bottleneck, float (offline zero-phase filter) | 0.860 |
| **Same, causal streaming-safe filter, float** | **0.825** |
| **Same, causal filter, INT8 (deployed model)** | **0.815** |

Deployed model: 14.4 KB INT8, about 0.52 M MACs per 2 s window, about 5.8 k parameters.

Deployment-level results (INT8, causal filter, k-of-m alarm rule, scored on 10 s segments):

| Preset | Threshold / rule | Segment recall | False alarms per hour |
|---|---|---|---|
| Balanced | val p95, alarm if 3 of last 5 windows flagged | 44.3% | 23.9 |
| Sensitive | val p90, alarm if 2 of last 3 windows flagged | 57.5% | 52.8 |

This is an anomaly-screening demo, not a clinical alarm. See `docs/results_summary.md` for limitations.

## Pipeline
| Step | Command | Output |
|---|---|---|
| Download data | `python scripts/download_data.py` | `data/mitdb/` (144 files) |
| Sanity plot | `python -m scripts.plot_record 100` | `results/record_100.png` |
| Train | `python -m src.train` | `models/autoencoder.keras` |
| Evaluate (float) | `python -m src.evaluate` | `results/float_metrics.json`, operating points, ROC |
| Reconstruction plot | `python -m scripts.plot_recon` | `results/recon_examples.png` |
| Quantize to INT8 | `python -m src.quantize` | `models/autoencoder_int8.tflite`, `ecg_ae_model.h` |
| Alarm-rule evaluation | `python -m src.alert_eval` | `results/alert_operating_points_int8.json` |
| Streaming simulation | `python -m src.stream_sim --record 210 --preset balanced --quiet` | console |
| Firmware test vectors | `python -m scripts.export_test_vectors` | `models/test_vectors.h` |

Run from the repo root. Full retrain order: train, evaluate, quantize, alert_eval.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Design choices
- MLII channel; causal 0.5-40 Hz Butterworth band-pass (`CAUSAL = True` in `src/config.py`); decimated to 180 Hz.
- 2 s windows (360 samples), 1 s hop; each window z-scored on its own.
- A window is normal only if every beat in it is annotated `N`; anything else is anomalous.
- Inter-patient split (de Chazal DS1/DS2, paced records excluded); validation records 215, 220, 223, 230.
- Thresholds are percentiles of reconstruction error on validation-normal windows, never tuned on test data.
- Decoder and encoder use only Conv/TransposeConv ops, which TFLM supports.

## Repo layout
`src/` pipeline code, `scripts/` utilities, `models/` trained models and C headers, `results/` metrics and plots,
`firmware/` ESP32-S3 code (next), `docs/` write-up notes.
