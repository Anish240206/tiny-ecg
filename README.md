# tinyecg

Self-supervised (reconstruction-based) ECG anomaly detection on an ultra-low-power edge node.
A small 1D-CNN autoencoder is trained on normal heartbeats only (MIT-BIH), quantized to INT8,
and run in a continuous sliding-window loop (target: ESP32-S3 + AD8232, TFLM).

Course project, BCSE306L (AI), VIT Vellore.

## Pipeline
| Step | Command | Output |
|---|---|---|
| 1. Download data | `python scripts/download_data.py` | `data/mitdb/` |
| 2. Sanity plot | `python -m scripts.plot_record 100` | `results/record_100.png` |
| 3. Train | `python -m src.train` | `models/autoencoder.keras`, `threshold_float.json` |
| 4. Evaluate (float) | `python -m src.evaluate` | `results/float_metrics.json`, ROC, histogram |
| 5. Quantize to INT8 | `python -m src.quantize` | `models/autoencoder_int8.tflite`, `ecg_ae_model.h` |
| 6. Streaming simulation | `python -m src.stream_sim --record 119` | console log |

Run everything from the repo root.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Design choices
- MLII channel, band-pass 0.5-40 Hz, decimated to 180 Hz, 2 s windows, hop 1 s (same as the live loop).
- Window label: normal only if every beat is `N`; anything else counts as anomalous.
- Inter-patient split (DS1 train/val, DS2 test), so no patient appears in both.
- Threshold = 99th percentile of reconstruction error on validation-normal windows.
