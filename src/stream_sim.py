"""Continuous sliding-window loop, simulated by replaying an MIT-BIH record sample by sample.
Exactly the logic the ESP32-S3 firmware will run:
  new sample -> ring buffer; every 1 s with a full 2 s window:
  z-score -> encode+decode (INT8) -> reconstruction error -> flag if > threshold
  -> alarm if >= k of the last m windows are flagged -> repeat forever.

Needs results/alert_operating_points_int8.json (written by `python -m src.alert_eval`).

    python -m src.stream_sim --record 210 --preset balanced --start 0 --seconds 300
"""
import argparse
import json
import time
from collections import deque
import numpy as np

from . import config as C
from .dataset import load_record, preprocess
from .tflite_utils import TFLiteAE

PRESETS = {"balanced": (95, 3, 5), "sensitive": (90, 2, 3)}   # (val percentile, k, m)


def get_threshold(pct, k, m):
    rows = json.load(open(C.RESULTS_DIR / "alert_operating_points_int8.json"))
    for r in rows:
        if r["val_percentile"] == pct and r["k"] == k and r["m"] == m:
            return r["threshold"]
    raise SystemExit("operating point not found; run `python -m src.alert_eval` first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=int, default=210)
    ap.add_argument("--preset", choices=PRESETS, default="balanced")
    ap.add_argument("--start", type=float, default=0, help="start offset in seconds")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--quiet", action="store_true", help="print only alarms and the summary")
    args = ap.parse_args()

    pct, k, m = PRESETS[args.preset]
    thr = get_threshold(pct, k, m)
    q = TFLiteAE(C.MODELS_DIR / "autoencoder_int8.tflite")

    sig, bs, sym = load_record(args.record)
    sig, bidx = preprocess(sig, bs)
    a = int(args.start * C.FS)
    sig = sig[a: a + int(args.seconds * C.FS)]
    bidx = bidx - a

    buf = deque(maxlen=C.WINDOW)
    recent = deque(maxlen=m)
    since_last = 0
    n_win = n_flag = n_alarm = n_episodes = n_truth_anom = 0
    prev_alarm = False
    lat = []
    print(f"record {args.record} | preset={args.preset} (val p{pct}, {k}-of-{m}) | threshold={thr:.4f}")
    for n, sample in enumerate(sig):                      # <- stands in for the ADC at 180 Hz
        buf.append(sample); since_last += 1
        if len(buf) < C.WINDOW or since_last < C.STRIDE_EVAL:
            continue
        since_last = 0
        w = np.array(buf, dtype=np.float32)
        w = ((w - w.mean()) / (w.std() + 1e-6))[:, None]
        t0 = time.perf_counter()
        e = q.error(w)
        lat.append((time.perf_counter() - t0) * 1e3)
        flag = e > thr
        recent.append(flag)
        alarm = sum(recent) >= k
        s0 = n + 1 - C.WINDOW
        in_win = sym[(bidx >= s0) & (bidx < n + 1)]
        truth = bool(np.any(in_win != C.NORMAL_SYMBOL))
        n_win += 1; n_flag += int(flag); n_alarm += int(alarm); n_truth_anom += int(truth)
        n_episodes += int(alarm and not prev_alarm); prev_alarm = alarm
        if not args.quiet or alarm:
            t = (a + n + 1) / C.FS
            print(f"t={t:7.1f}s err={e:.4f} {'flag ' if flag else '     '} "
                  f"{'*** ALARM ***' if alarm else '             '} truth={'ANOM' if truth else 'norm'}")
    print(f"\n{n_win} windows | truth-anomalous {n_truth_anom} | flagged {n_flag} | "
          f"alarm windows {n_alarm} | alarm episodes {n_episodes}")
    print(f"mean PC latency {np.mean(lat):.2f} ms (desktop Python, NOT the ESP32)")


if __name__ == "__main__":
    main()
