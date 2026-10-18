"""Continuous sliding-window loop, simulated by replaying an MIT-BIH record sample-by-sample.
This is exactly the logic the ESP32-S3 firmware will implement:
  buffer fills -> z-score -> encode+decode -> reconstruction error -> flag -> slide by 1 s -> repeat.

    python -m src.stream_sim --record 119
"""
import argparse
import json
import time
from collections import deque
import numpy as np

from . import config as C
from .dataset import load_record, preprocess
from .tflite_utils import TFLiteAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=int, default=119)
    ap.add_argument("--max-seconds", type=float, default=300)
    args = ap.parse_args()

    q = TFLiteAE(C.MODELS_DIR / "autoencoder_int8.tflite")
    thr = json.load(open(C.MODELS_DIR / "threshold_int8.json"))["threshold"]

    sig, bs, sym = load_record(args.record)
    sig, bidx = preprocess(sig, bs)
    sig = sig[: int(args.max_seconds * C.FS)]

    buf = deque(maxlen=C.WINDOW)
    since_last = 0
    flagged = total = 0
    lat = []
    print(f"record {args.record}: threshold={thr:.4f}")
    for n, sample in enumerate(sig):                 # <- stands in for the ADC at 180 Hz
        buf.append(sample); since_last += 1
        if len(buf) < C.WINDOW or since_last < C.STRIDE_EVAL:
            continue
        since_last = 0
        w = np.array(buf, dtype=np.float32)
        w = ((w - w.mean()) / (w.std() + 1e-6))[:, None]
        t0 = time.perf_counter()
        e = q.error(w)
        lat.append((time.perf_counter() - t0) * 1e3)
        start = n + 1 - C.WINDOW
        in_win = sym[(bidx >= start) & (bidx < n + 1)]
        truth = "ANOM" if np.any(in_win != C.NORMAL_SYMBOL) else "norm"
        is_anom = e > thr
        total += 1; flagged += int(is_anom)
        print(f"t={(n+1)/C.FS:7.1f}s  err={e:.4f}  {'ANOMALY' if is_anom else 'ok     '}  truth={truth}")
    print(f"\n{flagged}/{total} windows flagged | mean PC latency {np.mean(lat):.2f} ms (desktop, not ESP32)")


if __name__ == "__main__":
    main()
